#!/usr/bin/env python3
"""
MCP Server for semantic email search using LanceDB and fastembed.

Uses stdio transport for MCP communication.
Implements lazy initialization for heavy resources (embedder, vector store)
to ensure immediate MCP handshake response.

Configuration is read from config/config.toml (same as censor_agent).
"""
import asyncio
import argparse
import json

from pathlib import Path
from datetime import datetime
from typing import Any, AsyncIterator
from contextlib import asynccontextmanager

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

# Local imports
from src.config import load_config, Config
from src.embedder import Embedder
from src.vector_store import VectorStore



# Configuration - use absolute path based on script location
SCRIPT_DIR = Path(__file__).parent
CONFIG_FILE = SCRIPT_DIR / "config" / "config.toml"
LOG_FILE = SCRIPT_DIR / "mcp_server.log"


def parse_arguments():
    """Parse command line arguments for the MCP server."""
    parser = argparse.ArgumentParser(
        description="MCP Server for semantic email search using LanceDB and fastembed.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python mcp_server.py           # Run server without debug logging
  python mcp_server.py --debug   # Run server with debug logging to file
        """
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug logging to file (default: disabled)"
    )
    return parser.parse_args()


class FileLogger:
    """
    Logger that ONLY writes to file in debug mode.
    NEVER writes to stdout/stderr to avoid interfering with MCP stdio transport.
    """

    def __init__(self, is_debug: bool, log_file_path: str):
        self.is_debug = is_debug
        self.log_file_path = Path(log_file_path)

    def _write(self, level: str, msg: str) -> None:
        """Write a log entry to file with timestamp."""
        if not self.is_debug:
            return
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open(self.log_file_path, "a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] [{level}] {msg}\n")
        except Exception:
            pass

    def debug(self, msg: str) -> None:
        self._write("DEBUG", msg)

    def info(self, msg: str) -> None:
        self._write("INFO", msg)

    def warning(self, msg: str) -> None:
        self._write("WARNING", msg)

    def error(self, msg: str) -> None:
        self._write("ERROR", msg)


class ResourceManager:
    """
    Manages lazy initialization of heavy resources (embedder, vector store).
    Resources are initialized on first use, not at server startup.
    This allows the MCP handshake to complete immediately.
    """

    def __init__(self, config: Config, logger: FileLogger):
        self.config = config
        self.logger = logger
        self._embedder: Embedder | None = None
        self._vector_store: VectorStore | None = None
        self._init_lock = asyncio.Lock()
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    async def ensure_initialized(self) -> bool:
        """
        Ensure all resources are initialized.
        Trigger lazy initialization on first call.
        Thread-safe via asyncio.Lock.
        """
        if self._initialized:
            return True

        async with self._init_lock:
            if self._initialized:
                return True

            self.logger.info("Resources not ready, triggering lazy initialization...")

            try:
                # Initialize embedder
                self.logger.info("Initializing embedder...")
                self._embedder = Embedder(
                    model_name=self.config.indexing.embedding_model,
                    cache_dir=self.config.paths.fastembed_cache_path,
                    use_prefix=self.config.indexing.use_prefix,
                )
                self.logger.info("Embedder initialized")

                # Initialize vector store
                self.logger.info("Initializing vector store...")
                self._vector_store = VectorStore(
                    db_path=Path(self.config.paths.vector_db_path),
                    collection_name=self.config.indexing.collection_name,
                )
                self.logger.info("Vector store initialized")

                self.logger.info("All resources initialized successfully")
                return True

            except Exception as e:
                self.logger.error(f"Resource initialization failed: {e}")
                return False

    @property
    def embedder(self) -> Embedder | None:
        return self._embedder

    @property
    def vector_store(self) -> VectorStore | None:
        return self._vector_store

def validate_config(config: Config, logger: FileLogger) -> bool:
    """
    Validate configuration for MCP server.
    Only mandatory fields are validated - optional fields can be missing.
    """
    try:
        vector_db_path = Path(config.paths.vector_db_path)
        if not vector_db_path.exists():
            logger.error(f"vector_db_path does not exist: {vector_db_path}")
            return False
    except AttributeError:
        logger.error("paths.vector_db_path is missing")
        return False

    try:
        cache_path = Path(config.paths.fastembed_cache_path)
        if not cache_path.exists():
            logger.error(f"fastembed_cache_path does not exist: {cache_path}")
            return False
    except AttributeError:
        logger.error("paths.fastembed_cache_path is missing")
        return False

    try:
        chunk_size = config.indexing.chunk_size
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            logger.error(f"chunk_size must be integer > 0, got: {chunk_size}")
            return False

        chunk_overlap = config.indexing.chunk_overlap
        if not isinstance(chunk_overlap, int) or chunk_overlap < 0 or chunk_overlap >= chunk_size:
            logger.error(f"chunk_overlap must be integer >= 0 and < chunk_size ({chunk_size}), got: {chunk_overlap}")
            return False

        embedding_model = config.indexing.embedding_model
        if not isinstance(embedding_model, str) or not embedding_model.strip():
            logger.error(f"embedding_model must be non-empty string, got: {embedding_model}")
            return False

        collection_name = config.indexing.collection_name
        if not isinstance(collection_name, str) or not collection_name.strip():
            logger.error(f"collection_name must be non-empty string, got: {collection_name}")
            return False

        use_prefix = config.indexing.use_prefix
        if not isinstance(use_prefix, bool):
            logger.error(f"use_prefix must be bool, got: {use_prefix}")
            return False
    except AttributeError as e:
        logger.error(f"indexing section missing mandatory field: {e}")
        return False

    try:
        top_n = config.search.top_n
        if not isinstance(top_n, int) or top_n <= 0:
            logger.error(f"top_n must be integer > 0, got: {top_n}")
            return False
    except AttributeError:
        logger.error("search.top_n is missing")
        return False

    logger.info("Configuration validation passed")
    return True


def get_tools() -> list[Tool]:
    """Define and return all MCP tools exposed by this server."""
    return [
        Tool(
            name="search_emails",
            description=(
                "Perform a semantic search over indexed emails. "
                "Returns the top matching email chunks with metadata."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query text to find semantically similar emails",
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Number of results to return (optional, defaults to config value)",
                        "minimum": 1,
                        "maximum": 100,
                    },
                    "filter": {
                        "type": "string",
                        "description": "Optional LanceDB filter expression (e.g., 'label = \"work\"')",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="get_system_status",
            description="Returns indexing statistics and key configuration parameters.",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


async def handle_search_emails(
    query: str,
    top_n: int | None,
    filter_expr: str | None,
    config: Config,
    resource_manager: ResourceManager,
    logger: FileLogger,
) -> list[TextContent]:
    """
    Handle the search_emails tool call.
    Triggers lazy initialization if resources are not ready.
    """
    is_ready = await resource_manager.ensure_initialized()
    if not is_ready:
        return [TextContent(type="text", text=json.dumps({
            "error": "Failed to initialize search resources"
        }))]

    query = query.strip()
    if not query:
        return [TextContent(type="text", text=json.dumps({
            "error": "Empty query"
        }))]

    try:
        embedder = resource_manager.embedder
        embedding = embedder.embed_query(query)
    except Exception as e:
        logger.error(f"Failed to embed query: {e}")
        return [TextContent(type="text", text=json.dumps({
            "error": f"Failed to process query: {e}"
        }))]

    if top_n is None:
        top_n = config.search.top_n

    try:
        vector_store = resource_manager.vector_store
        # Use hybrid search combining vector similarity and full-text search
        # This enables matching queries against both subject (FTS) and body (vector)
        results = vector_store.hybrid_search(query, embedding, top_n, filter_expr)
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return [TextContent(type="text", text=json.dumps({
            "error": f"Search failed: {e}"
        }))]

    results_list = []
    for result in results:
        results_list.append({
            "message_id": result.message_id,
            "subject": result.subject,
            "from_address": result.from_address,
            "to_addresses": result.to_addresses,
            "date_iso": result.date_iso,
            "content": result.content
        })

    response = {
        "query": query,
        "total_results": len(results_list),
        "results": results_list
    }

    return [TextContent(type="text", text=json.dumps(response, indent=2))]


async def handle_get_system_status(
    config: Config,
    resource_manager: ResourceManager,
    logger: FileLogger,
) -> list[TextContent]:
    """
    Handle the get_system_status tool call.
    Returns statistics and configuration info.
    """
    is_initialized = resource_manager.is_initialized

    if is_initialized:
        status = "ready"
        indexed_chunks = 0
        disk_size_bytes = 0

        try:
            vector_store = resource_manager.vector_store
            indexed_chunks = vector_store.count()
            disk_size_bytes = vector_store.disk_size_bytes()
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            status = "error"
    else:
        status = "initializing"
        indexed_chunks = 0
        disk_size_bytes = 0

    config_data = {
        "embedding_model": config.indexing.embedding_model,
        "collection_name": config.indexing.collection_name,
        "chunk_size": config.indexing.chunk_size,
        "top_n": config.search.top_n
    }

    response = {
        "status": status,
        "initialized": is_initialized,
        "indexed_chunks": indexed_chunks,
        "disk_size_bytes": disk_size_bytes,
        "config": config_data
    }

    return [TextContent(type="text", text=json.dumps(response, indent=2))]


@asynccontextmanager
async def app_lifespan(server: Server, is_debug: bool = False) -> AsyncIterator[dict]:
    """
    Manage application lifecycle.

    IMPORTANT: Resources are NOT initialized here.
    They are initialized lazily on first tool call.
    This allows the MCP handshake to complete immediately.
    """
    logger = FileLogger(is_debug=is_debug, log_file_path=LOG_FILE)

    logger.info("=" * 60)
    logger.info("MCP Email Search Server starting...")
    logger.info("Resources will be initialized on first use (lazy pattern)")
    logger.info("=" * 60)

    # Load configuration
    config_path = Path(CONFIG_FILE)
    if not config_path.exists():
        logger.error(f"Configuration file not found: {CONFIG_FILE}")
        yield {"error": "Config not found", "logger": logger}
        return

    try:
        config = load_config(CONFIG_FILE)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        yield {"error": f"Config load failed: {e}", "logger": logger}
        return

    # Validate configuration
    if not validate_config(config, logger):
        logger.error("Configuration validation failed")
        yield {"error": "Config validation failed", "logger": logger}
        return

    logger.info("Configuration loaded and validated")

    # Create resource manager (does NOT initialize resources yet)
    resource_manager = ResourceManager(config=config, logger=logger)

    logger.info("Server ready, waiting for MCP client connection...")

    yield {
        "config": config,
        "resource_manager": resource_manager,
        "logger": logger,
    }

    logger.info("MCP Email Search Server shutting down...")


async def main():
    """Main entry point for the MCP server."""
    args = parse_arguments()
    
    server = Server(
        name="email-search-mcp",
        version="1.0.0",
        lifespan=lambda server: app_lifespan(server, args.debug),
    )

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return get_tools()

    @server.call_tool()
    async def call_tool(
        name: str,
        arguments: dict[str, Any]
    ) -> list[TextContent]:
        context = server.request_context
        lifespan_context = context.lifespan_context

        # Check for initialization errors
        if "error" in lifespan_context:
            return [TextContent(type="text", text=json.dumps({
                "error": f"Server initialization failed: {lifespan_context['error']}"
            }))]

        config = lifespan_context["config"]
        resource_manager = lifespan_context["resource_manager"]
        logger = lifespan_context["logger"]

        if name == "search_emails":
            query = arguments.get("query", "")
            top_n = arguments.get("top_n")
            filter_expr = arguments.get("filter")
            return await handle_search_emails(
                query=query,
                top_n=top_n,
                filter_expr=filter_expr,
                config=config,
                resource_manager=resource_manager,
                logger=logger,
            )

        elif name == "get_system_status":
            return await handle_get_system_status(
                config=config,
                resource_manager=resource_manager,
                logger=logger,
            )

        else:
            return [TextContent(type="text", text=json.dumps({
                "error": f"Unknown tool: {name}"
            }))]

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
