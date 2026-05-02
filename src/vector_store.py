import os
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import lancedb
import pyarrow as pa
from lance.dataset import AutoCleanupConfig

# Auto-cleanup periodically removes old LanceDB version manifests during write
# operations, preventing unbounded disk growth. Every AUTO_CLEANUP_INTERVAL
# commits, all versions older than AUTO_CLEANUP_OLDER_THAN_SECONDS are deleted.
AUTO_CLEANUP_INTERVAL = 20
AUTO_CLEANUP_OLDER_THAN_SECONDS = 0

# Hybrid search configuration: pool multiplier for RRF fusion
# Each search (vector + FTS) retrieves top_n * HYBRID_POOL_MULTIPLIER candidates
# before fusion. Higher values improve recall at the cost of performance.
HYBRID_POOL_MULTIPLIER = 2


@dataclass
class ChunkRecord:
    chunk_id: str
    text: str
    embedding: list[float]
    metadata: dict[str, Any]


@dataclass
class SearchResult:
    content: str
    subject: str
    date_iso: str
    from_address: str
    to_addresses: str
    message_id: str


class VectorStore:
    def __init__(self, db_path: Path, collection_name: str) -> None:
        self._db_path = db_path
        self._table_name = collection_name

        db_path.parent.mkdir(parents=True, exist_ok=True)

        self._db = lancedb.connect(str(db_path))

        try:
            self._table = self._db.open_table(collection_name)
        except (FileNotFoundError, ValueError):
            self._table = None

        self._enable_auto_cleanup()

    def _enable_auto_cleanup(self) -> None:
        """Enable LanceDB auto-cleanup on the underlying dataset.

        Configures the Lance dataset to automatically remove old version
        manifests during write operations. The settings persist in the
        dataset manifest and survive across sessions.
        """
        if self._table is None:
            return
        lance_ds = self._table.to_lance()
        lance_ds.optimize.enable_auto_cleanup(
            AutoCleanupConfig(
                interval=AUTO_CLEANUP_INTERVAL,
                older_than_seconds=AUTO_CLEANUP_OLDER_THAN_SECONDS,
            )
        )

    def _ensure_fts_index(self) -> None:
        """Create Full-Text Search index on subject and text columns if not exists.

        LanceDB FTS requires tantivy and creates an inverted index for text search.
        This enables keyword matching on email subjects and content chunks.

        For new tables, this is called automatically in _create_table().
        For existing tables without FTS index, hybrid_search() calls this as fallback.
        """
        if self._table is None:
            return

        # Check if FTS indexes already exist to avoid recreation
        lance_ds = self._table.to_lance()
        existing_indices = lance_ds.list_indices()
        # LanceDB creates "Inverted" type indices for FTS
        # Each index has a "fields" list with column names; extract first field from each FTS index
        existing_fts_columns = {idx.get("fields", [None])[0] for idx in existing_indices if idx.get("type") == "Inverted"}

        # Create FTS index for subject column
        if "subject" not in existing_fts_columns:
            self._table.create_fts_index("subject")

        # Create FTS index for text column
        if "text" not in existing_fts_columns:
            self._table.create_fts_index("text")

    def _create_table(self, records: list[ChunkRecord]) -> None:
        if not records:
            return

        data = {
            "chunk_id": [r.chunk_id for r in records],
            "text": [r.text for r in records],
            "vector": [r.embedding for r in records],
            "message_id": [r.metadata.get("message_id", "") for r in records],
            "subject": [r.metadata.get("subject", "") for r in records],
            "date_ts": [r.metadata.get("date_ts", 0) for r in records],
            "date_iso": [r.metadata.get("date_iso", "") for r in records],
            "from_address": [r.metadata.get("from_address", "") for r in records],
            "from_domain": [r.metadata.get("from_domain", "") for r in records],
            "to_addresses": [r.metadata.get("to_addresses", "") for r in records],
            "to_domains": [r.metadata.get("to_domains", "") for r in records],
            "cc_addresses": [r.metadata.get("cc_addresses", "") for r in records],
            "account": [r.metadata.get("account", "") for r in records],
            "label": [r.metadata.get("label", "") for r in records],
            "chunk_index": [r.metadata.get("chunk_index", 0) for r in records],
            "chunk_total": [r.metadata.get("chunk_total", 0) for r in records],
            "mbox_file": [r.metadata.get("mbox_file", "") for r in records],
        }

        arrow_table = pa.table(data)

        self._table = self._db.create_table(
            self._table_name,
            data=arrow_table,
            mode="create"
        )
        self._enable_auto_cleanup()

        # Create FTS index immediately for fresh tables
        # This avoids lazy index creation during first hybrid_search query
        self._ensure_fts_index()

    def upsert(self, records: list[ChunkRecord]) -> None:
        if not records:
            return

        if self._table is None:
            self._create_table(records)
            return

        data = {
            "chunk_id": [r.chunk_id for r in records],
            "text": [r.text for r in records],
            "vector": [r.embedding for r in records],
            "message_id": [r.metadata.get("message_id", "") for r in records],
            "subject": [r.metadata.get("subject", "") for r in records],
            "date_ts": [r.metadata.get("date_ts", 0) for r in records],
            "date_iso": [r.metadata.get("date_iso", "") for r in records],
            "from_address": [r.metadata.get("from_address", "") for r in records],
            "from_domain": [r.metadata.get("from_domain", "") for r in records],
            "to_addresses": [r.metadata.get("to_addresses", "") for r in records],
            "to_domains": [r.metadata.get("to_domains", "") for r in records],
            "cc_addresses": [r.metadata.get("cc_addresses", "") for r in records],
            "account": [r.metadata.get("account", "") for r in records],
            "label": [r.metadata.get("label", "") for r in records],
            "chunk_index": [r.metadata.get("chunk_index", 0) for r in records],
            "chunk_total": [r.metadata.get("chunk_total", 0) for r in records],
            "mbox_file": [r.metadata.get("mbox_file", "") for r in records],
        }

        arrow_table = pa.table(data)

        # LanceDB merge_insert implements an upsert pattern:
        # - "chunk_id" is the merge key used to match existing records
        # - when_matched_update_all(): Updates all columns if chunk_id exists
        # - when_not_matched_insert_all(): Inserts new record if chunk_id not found
        # This ensures idempotent writes - re-indexing updates instead of duplicates
        self._table.merge_insert("chunk_id") \
            .when_matched_update_all() \
            .when_not_matched_insert_all() \
            .execute(arrow_table)

    def search(
        self, query_embedding: list[float], top_n: int, lance_filter: str | None
    ) -> list[SearchResult]:
        if self._table is None:
            return []

        # Semantic search: LanceDB computes cosine similarity between query embedding
        # and stored vectors, returning the most semantically similar chunks.
        # Cosine similarity ranges from -1 (opposite) to 1 (identical).
        query = self._table.search(query_embedding).limit(top_n)

        # Optional metadata filter using SQL-like syntax on indexed columns.
        # Example: lance_filter = "from_address = 'user@example.com' AND date_ts > 1704067200"
        # Available columns: message_id, subject, date_ts, date_iso, from_address,
        # from_domain, to_addresses, to_domains, account, label, etc.
        if lance_filter:
            query = query.where(lance_filter)

        results_table = query.to_arrow()

        if results_table.num_rows == 0:
            return []

        results_dict = results_table.to_pydict()

        results = []
        for i in range(len(results_dict["text"])):
            results.append(SearchResult(
                content=results_dict["text"][i],
                subject=results_dict["subject"][i],
                date_iso=results_dict["date_iso"][i],
                from_address=results_dict["from_address"][i],
                to_addresses=results_dict["to_addresses"][i],
                message_id=results_dict["message_id"][i],
            ))

        return results

    def hybrid_search(
        self,
        query_text: str,
        query_embedding: list[float],
        top_n: int,
        lance_filter: str | None
    ) -> list[SearchResult]:
        """Perform hybrid search combining vector similarity and full-text search.

        Algorithm:
        1. Execute vector search and FTS search in parallel (conceptually)
        2. Retrieve top_k = top_n * HYBRID_POOL_MULTIPLIER from each
        3. Apply filter BEFORE fusion (filter on candidate pool)
        4. Merge using Reciprocal Rank Fusion (RRF): score = sum(1/(k+rank))
        5. Return top_n results sorted by RRF score

        Args:
            query_text: Raw query text for FTS search
            query_embedding: Query embedding for vector search
            top_n: Number of final results to return
            lance_filter: Optional LanceDB filter expression (applied to both searches)

        Returns:
            List of SearchResult sorted by RRF relevance score
        """
        if self._table is None:
            return []

        # Ensure FTS index exists
        self._ensure_fts_index()

        pool_size = top_n * HYBRID_POOL_MULTIPLIER

        # Vector search with filter applied
        try:
            vector_query = self._table.search(query_embedding).limit(pool_size)
            if lance_filter:
                vector_query = vector_query.where(lance_filter)
            vector_results = vector_query.to_arrow()
        except Exception:
            vector_results = None

        # FTS search with filter applied
        try:
            fts_query = self._table.search(query_text, query_type="fts").limit(pool_size)
            if lance_filter:
                fts_query = fts_query.where(lance_filter)
            fts_results = fts_query.to_arrow()
        except Exception:
            fts_results = None

        # If one search fails, return results from the other
        if vector_results is None and fts_results is None:
            return []
        elif vector_results is None:
            return self._results_from_arrow(fts_results, top_n)
        elif fts_results is None:
            return self._results_from_arrow(vector_results, top_n)

        # RRF Fusion
        k = 60  # RRF constant
        rrf_scores = {}

        # Process vector results (rank starts at 1)
        vector_dict = vector_results.to_pydict()
        for rank, chunk_id in enumerate(vector_dict.get("chunk_id", []), 1):
            if chunk_id not in rrf_scores:
                rrf_scores[chunk_id] = {
                    "score": 0,
                    "data": {key: vector_dict[key][rank-1] for key in vector_dict}
                }
            rrf_scores[chunk_id]["score"] += 1.0 / (k + rank)

        # Process FTS results
        fts_dict = fts_results.to_pydict()
        for rank, chunk_id in enumerate(fts_dict.get("chunk_id", []), 1):
            if chunk_id not in rrf_scores:
                rrf_scores[chunk_id] = {
                    "score": 0,
                    "data": {key: fts_dict[key][rank-1] for key in fts_dict}
                }
            rrf_scores[chunk_id]["score"] += 1.0 / (k + rank)

        # Sort by RRF score and return top_n
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1]["score"], reverse=True)

        results = []
        for chunk_id, item in sorted_results[:top_n]:
            data = item["data"]
            results.append(SearchResult(
                content=data.get("text", ""),
                subject=data.get("subject", ""),
                date_iso=data.get("date_iso", ""),
                from_address=data.get("from_address", ""),
                to_addresses=data.get("to_addresses", ""),
                message_id=data.get("message_id", ""),
            ))

        return results

    def _results_from_arrow(self, arrow_table, limit: int) -> list[SearchResult]:
        """Convert Arrow table to list of SearchResult (fallback helper)."""
        if arrow_table is None or arrow_table.num_rows == 0:
            return []

        results_dict = arrow_table.to_pydict()
        results = []
        for i in range(min(limit, len(results_dict["text"]))):
            results.append(SearchResult(
                content=results_dict["text"][i],
                subject=results_dict["subject"][i],
                date_iso=results_dict["date_iso"][i],
                from_address=results_dict["from_address"][i],
                to_addresses=results_dict["to_addresses"][i],
                message_id=results_dict["message_id"][i],
            ))
        return results

    def count(self) -> int:
        if self._table is None:
            return 0
        return self._table.count_rows()

    def disk_size_bytes(self) -> int:
        total = 0
        for root, _dirs, files in os.walk(self._db_path):
            for f in files:
                total += os.path.getsize(os.path.join(root, f))
        return total

    def optimize(self) -> None:
        """Compact data files and remove all old versions from the vector store.

        Merges small fragment files into larger ones and deletes all version
        manifests except the latest. Should be called after bulk indexing to
        reclaim disk space. Uses delete_unverified=True because censor_agent
        is the sole writer process.
        """
        if self._table is None:
            return
        self._table.optimize(
            cleanup_older_than=timedelta(0),
            delete_unverified=True,
        )