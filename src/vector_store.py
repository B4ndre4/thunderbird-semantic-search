import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lancedb
import pyarrow as pa


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