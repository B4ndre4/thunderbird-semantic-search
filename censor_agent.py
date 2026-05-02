import argparse
import sys
from pathlib import Path
from datetime import datetime, timezone

from src.config import load_config, Config
from src.state_db import StateDB
from src.ollama_provider import OllamaProvider
from src.llamacpp_provider import LlamaCppProvider
from src.embedder import Embedder
from src.vector_store import VectorStore, ChunkRecord
from src.llm_provider import LLMProvider
from src.mbox_parser import parse_mbox, count_remaining_emails, get_email_by_message_id
from src.chunker import chunk_text
import signal
import threading

# Global shutdown event for graceful termination
shutdown_event = threading.Event()

# Use absolute paths based on script location
SCRIPT_DIR = Path(__file__).parent
log_file = SCRIPT_DIR / "censor_agent.log"
config_file = SCRIPT_DIR / "config" / "config.toml"

class Logger:
    """Handles logging to console and file based on script parameters.

    Console output respects --silent for non-error messages.
    File logging is enabled only when --debug is True.
    Error messages always go to stderr (ignore --silent).
    """

    def __init__(self, is_silent: bool, is_debug: bool, log_file_path: str):
        self.is_silent = is_silent
        self.is_debug = is_debug
        self.log_file_path = log_file_path

    def _write_to_file(self, msg: str) -> None:
        """Write a message to the log file with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file_path, "a") as f:
            f.write(f"[{timestamp}] {msg}\n")

    def console(self, msg: str) -> None:
        """Print message to stdout if not in silent mode."""
        if not self.is_silent:
            print(msg)

    def error(self, msg: str) -> None:
        """Print error message to stderr (always, ignores --silent)."""
        print(msg, file=sys.stderr)

    def debug(self, msg: str) -> None:
        """Write message to log file if debug mode is enabled."""
        if self.is_debug:
            self._write_to_file(msg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Censor Agent — index and classify Thunderbird emails using LLM"
    )
    parser.add_argument(
        "--file", type=str, default=None,
        help="Path to a single mbox file for backlog mode (can be relative to CWD)"
    )
    parser.add_argument(
        "--silent", action="store_true", default=False,
        help="Disable all console output"
    )
    parser.add_argument(
        "--retry", type=str, default=None,
        help="Retry processing failed emails for the specified mbox file"
    )
    parser.add_argument(
        "--debug", action="store_true", default=False,
        help="Write verbose output to logs/censor_agent.log"
    )
    args = parser.parse_args()

    # If --file is passed we must check the specified path existence
    if args.file:
        file_path = Path(args.file)
        if not file_path.is_absolute():
            file_path = Path.cwd() / file_path
        if not file_path.exists():
            print(f"ERROR: --file does not exist: {file_path}", file=sys.stderr)
            sys.exit(1)
        args.file = str(file_path)

    # If --retry is passed we must check the specified path existence
    if args.retry:
        retry_path = Path(args.retry)
        if not retry_path.is_absolute():
            retry_path = Path.cwd() / retry_path
        if not retry_path.exists():
            print(f"ERROR: --retry file does not exist: {retry_path}", file=sys.stderr)
            sys.exit(1)
        args.retry = str(retry_path)

    return args


def config_is_valid(config, logger: Logger) -> bool:
    if not isinstance(config.search.top_n, int) or config.search.top_n <= 0:
        logger.error("ERROR: [search] top_n must be an integer greater than 0")
        return False

    if not isinstance(config.indexing.chunk_size, int) or config.indexing.chunk_size <= 0:
        logger.error("ERROR: [indexing] chunk_size must be an integer greater than 0")
        return False

    if not isinstance(config.indexing.chunk_overlap, int) or config.indexing.chunk_overlap < 0:
        logger.error("ERROR: [indexing] chunk_overlap must be an integer greater than or equal to 0")
        return False

    if config.indexing.chunk_overlap >= config.indexing.chunk_size:
        logger.error(f"ERROR: [indexing] chunk_overlap ({config.indexing.chunk_overlap}) must be less than chunk_size ({config.indexing.chunk_size})")
        return False

    if not isinstance(config.indexing.embedding_model, str) or not config.indexing.embedding_model.strip():
        logger.error("ERROR: [indexing] embedding_model must be a non-empty string")
        return False

    if not isinstance(config.indexing.collection_name, str) or not config.indexing.collection_name.strip():
        logger.error("ERROR: [indexing] collection_name must be a non-empty string")
        return False

    if not isinstance(config.indexing.use_prefix, bool):
        logger.error("ERROR: [indexing] use_prefix must be a boolean")
        return False

    if not Path(config.paths.fastembed_cache_path).exists():
        logger.error(f"ERROR: [paths] fastembed_cache_path does not exist: {config.paths.fastembed_cache_path}")
        return False

    if not Path(config.paths.state_db_path).parent.exists():
        logger.error(f"ERROR: [paths] state_db_path parent directory does not exist: {Path(config.paths.state_db_path).parent}")
        return False

    if not Path(config.paths.vector_db_path).parent.exists():
        logger.error(f"ERROR: [paths] vector_db_path parent directory does not exist: {Path(config.paths.vector_db_path).parent}")
        return False

    # Note: mbox_paths validation skipped here because it requires args
    # This check is performed separately in main() when needed

    valid_providers = {"ollama", "ollama cloud", "llama.cpp"}
    if not isinstance(config.llm.provider, str) or config.llm.provider not in valid_providers:
        logger.error(f"ERROR: [llm] provider must be one of: {', '.join(valid_providers)}")
        return False

    if config.llm.provider == "ollama":
        if not isinstance(config.llm.ollama_local_url, str) or not config.llm.ollama_local_url.strip():
            logger.error("ERROR: [llm] ollama_local_url must be a non-empty string when provider is 'ollama'")
            return False
        if not isinstance(config.llm.ollama_model, str) or not config.llm.ollama_model.strip():
            logger.error("ERROR: [llm] ollama_model must be a non-empty string when provider is 'ollama'")
            return False

    if config.llm.provider == "ollama cloud":
        if not isinstance(config.llm.ollama_model, str) or not config.llm.ollama_model.strip():
            logger.error("ERROR: [llm] ollama_model must be a non-empty string when provider is 'ollama cloud'")
            return False
        if not isinstance(config.llm.ollama_api_key, str) or not config.llm.ollama_api_key.strip():
            logger.error("ERROR: [llm] ollama_api_key must be a non-empty string when provider is 'ollama cloud'")
            return False

    if config.llm.provider == "llama.cpp":
        if not isinstance(config.llm.llamacpp_model_path, str) or not config.llm.llamacpp_model_path.strip():
            logger.error("ERROR: [llm] llamacpp_model_path must be a non-empty string when provider is 'llama.cpp'")
            return False

    for prompt_name in ["censor_prompt", "censor_extract_prompt", "censor_clean_prompt"]:
        prompt_value = getattr(config.llm, prompt_name)
        if not isinstance(prompt_value, str) or not prompt_value.strip():
            logger.error(f"ERROR: [llm] {prompt_name} must be a non-empty string")
            return False

    # Validate email_patterns section
    if not hasattr(config, 'email_patterns'):
        logger.error("ERROR: [email_patterns] section is missing from config")
        return False

    required_pattern_keys = ["forward_subject", "forward_body", "thread_reply"]
    for key in required_pattern_keys:
        if not hasattr(config.email_patterns, key):
            logger.error(f"ERROR: [email_patterns] missing required key '{key}'")
            return False
        pattern_list = getattr(config.email_patterns, key)
        if not isinstance(pattern_list, list) or len(pattern_list) == 0:
            logger.error(f"ERROR: [email_patterns] {key} must be a non-empty list")
            return False

    return True


def validate_mbox_paths(config, args, logger: Logger) -> bool:
    """Validate mbox_paths when neither --file nor --retry is specified."""
    if args.file or args.retry:
        return True

    if not config.paths.mbox_paths or len(config.paths.mbox_paths) == 0:
        logger.error("ERROR: [paths] mbox_paths must contain at least one path when --file is not specified")
        return False

    for mbox_path in config.paths.mbox_paths:
        if not Path(mbox_path).exists():
            logger.error(f"ERROR: [paths] mbox_paths entry does not exist: {mbox_path}")
            return False

    return True

def process_file(
    mbox_path: Path,
    config: Config,
    state_db: StateDB,
    llm_provider: LLMProvider,
    embedder: Embedder,
    vector_store: VectorStore,
    logger: Logger,
) -> dict[str, int]:
    processed = 0
    skipped = 0
    discarded = 0
    indexed = 0
    errors = 0

    last_id = state_db.get_last_message_id(mbox_path.name)
    remaining = count_remaining_emails(mbox_path, last_id)

    # Resume mechanism: retrieves the last processed message_id from state database
    # to enable incremental processing. This allows the agent to skip already
    # processed emails and continue from where it left off in case of interruptions.

    logger.console(f"Processing file: {mbox_path.name}, {remaining} emails remaining")
    logger.debug(f"Processing file: {mbox_path.name}, remaining from message_id: {last_id}")

    for email in parse_mbox(mbox_path, last_id, config):
        processed += 1

        # Check for graceful shutdown request
        if shutdown_event.is_set():
            logger.console("\nShutdown requested. Completing current email and exiting...")
            logger.debug("Shutdown requested during processing")
            break

        subject_preview = email.subject[:25] if email.subject else "(no subject)"
        logger.console(f"Processing email {processed}/{remaining}: {subject_preview}... ({email.date_iso})")
        logger.debug(f"Processing email {processed}: message_id={email.message_id}, subject={email.subject}, date={email.date_iso}")

        # Deduplication mechanism: uses SHA-256 hash of email body to detect duplicates.
        # Note: This approach considers emails with identical body content as duplicates,
        # even if subjects differ. This is intentional to handle forwarded messages or
        # multiple replies with quoted text. However, it may miss near-duplicates with
        # minor formatting changes.
        if state_db.hash_exists(email.body_hash):
            skipped += 1
            state_db.set_last_message_id(mbox_path.name, email.message_id)
            logger.debug("Result: skipped (duplicate hash)")
            continue

        classification_prompt = config.llm.censor_prompt.format(
            subject=email.subject or "",
            from_address=email.from_address or "",
            date_iso=email.date_iso or "",
            body=email.body or ""
        )

        try:
            raw_response = llm_provider.get_response(classification_prompt)
        except Exception as e:
            state_db.add_failed_email(mbox_path.name, email.message_id, email.subject, f"Classification error: {e}")
            errors += 1
            state_db.set_last_message_id(mbox_path.name, email.message_id)
            logger.debug(f"Result: error (classification failed: {e})")
            continue

        # Multi-language classification mapping: extracts first token from LLM response
        # and maps to standardized labels using keywords in Italian (lavoro/privata/mista),
        # English (work/private/mixed), Spanish (trabajo/privado/mixto), German
        # (arbeit/privat/gemischt), and French (travail/privée/mixte). This supports
        # multilingual LLM outputs while maintaining consistent internal label system.
        first_word = raw_response.strip().split()[0].lower() if raw_response.strip() else ""

        label = None
        if first_word in ("lavoro", "work", "trabajo", "arbeit", "travail"):
            label = "work"
        elif first_word in ("privata", "private", "privado", "privat", "privée"):
            label = "private"
        elif first_word in ("mista", "mixed", "mixto", "gemischt", "mixte"):
            label = "mixed"
        else:
            state_db.add_failed_email(mbox_path.name, email.message_id, email.subject, f"Unknown classification: {first_word}")
            errors += 1
            state_db.set_last_message_id(mbox_path.name, email.message_id)
            logger.debug(f"Result: error (unknown classification: {first_word})")
            continue

        if label == "private":
            discarded += 1
            state_db.set_last_message_id(mbox_path.name, email.message_id)
            logger.debug("Result: discarded (private)")
            continue

        # Content extraction prompt selection: censor_clean_prompt is used for "work" emails
        # (already work-focused, minor cleanup needed), while censor_extract_prompt is used
        # for "mixed" emails (requires extraction of work-relevant portions, filtering out
        # personal/social content). Both prompts are defined in config and customized for
        # their specific extraction tasks.
        if label == "work":
            extract_prompt = config.llm.censor_clean_prompt
        else:
            extract_prompt = config.llm.censor_extract_prompt

        extraction_prompt = extract_prompt.format(
            subject=email.subject or "",
            from_address=email.from_address or "",
            body=email.body or ""
        )

        try:
            work_text = llm_provider.get_response(extraction_prompt)
        except Exception as e:
            state_db.add_failed_email(mbox_path.name, email.message_id, email.subject, f"Extraction error: {e}")
            errors += 1
            state_db.set_last_message_id(mbox_path.name, email.message_id)
            logger.debug(f"Result: error (extraction failed: {e})")
            continue

        if not work_text or not work_text.strip():
            discarded += 1
            state_db.set_last_message_id(mbox_path.name, email.message_id)
            logger.debug(f"Result: discarded ({label} with empty content)")
            continue

        try:
            chunks = chunk_text(work_text, config.indexing.chunk_size, config.indexing.chunk_overlap)
        except Exception as e:
            state_db.add_failed_email(mbox_path.name, email.message_id, email.subject, f"Chunking error: {e}")
            errors += 1
            state_db.set_last_message_id(mbox_path.name, email.message_id)
            logger.debug(f"Result: error (chunking failed: {e})")
            continue

        records = []
        try:
            for chunk in chunks:
                # Include subject in embedding text to strengthen semantic signal
                # Format: "Subject: {subject}\n\n{chunk_text}"
                # This ensures vector search can match queries against both subject and body
                text_with_subject = f"Subject: {email.subject}\n\n{chunk.text}"
                embedding = embedder.embed_passage(text_with_subject)
                chunk_id = f"{email.message_id}__{chunk.chunk_index}"
                metadata = {
                    "message_id": email.message_id,
                    "subject": email.subject,
                    "date_ts": email.date_ts,
                    "date_iso": email.date_iso,
                    "from_address": email.from_address,
                    "from_domain": email.from_domain,
                    "to_addresses": email.to_addresses,
                    "to_domains": email.to_domains,
                    "cc_addresses": email.cc_addresses,
                    "account": email.account,
                    "label": label,
                    "chunk_index": chunk.chunk_index,
                    "chunk_total": chunk.chunk_total,
                    "mbox_file": mbox_path.name,
                }
                records.append(ChunkRecord(
                    chunk_id=chunk_id,
                    text=text_with_subject,  # Include subject for consistency with embedding
                    embedding=embedding,
                    metadata=metadata,
                ))
        except Exception as e:
            state_db.add_failed_email(mbox_path.name, email.message_id, email.subject, f"Embedding error: {e}")
            errors += 1
            state_db.set_last_message_id(mbox_path.name, email.message_id)
            logger.debug(f"Result: error (embedding failed: {e})")
            continue

        try:
            vector_store.upsert(records)
        except Exception as e:
            state_db.add_failed_email(mbox_path.name, email.message_id, email.subject, f"Vector store error: {e}")
            errors += 1
            state_db.set_last_message_id(mbox_path.name, email.message_id)
            logger.debug(f"Result: error (vector store failed: {e})")
            continue

        state_db.add_hash(email.body_hash)
        state_db.set_last_message_id(mbox_path.name, email.message_id)
        indexed += 1
        logger.debug(f"Result: indexed ({label}, {len(chunks)} chunks)")

    logger.console(f"Completed: {mbox_path.name}")
    logger.console(f"  Processed: {processed} | Indexed: {indexed} | Skipped: {skipped} | Discarded: {discarded} | Errors: {errors}")
    logger.debug(f"Completed processing file: {mbox_path.name}, stats: processed={processed}, skipped={skipped}, discarded={discarded}, indexed={indexed}, errors={errors}")

    return {"processed": processed, "skipped": skipped, "discarded": discarded, "indexed": indexed, "errors": errors}

def process_retries(
    mbox_path: Path,
    config: Config,
    state_db: StateDB,
    llm_provider: LLMProvider,
    embedder: Embedder,
    vector_store: VectorStore,
    logger: Logger,
) -> dict[str, int]:
    processed = 0
    skipped = 0
    discarded = 0
    indexed = 0
    errors = 0

    failed_emails = state_db.get_failed_emails(mbox_path.name)

    if not failed_emails:
        logger.console(f"No failed emails to retry for: {mbox_path.name}")
        logger.debug(f"No failed emails to retry for: {mbox_path.name}")
        return {"processed": 0, "skipped": 0, "discarded": 0, "indexed": 0, "errors": 0}

    logger.console(f"Retrying {len(failed_emails)} failed emails from: {mbox_path.name}")
    logger.debug(f"Retrying {len(failed_emails)} failed emails from: {mbox_path.name}")

    # Create a set of message_ids to search for
    failed_ids_set = {f[0] for f in failed_emails}

    # Dictionary to store found emails
    found_emails_dict = {}

    logger.console(f"Scanning mbox to find {len(failed_emails)} failed emails...")
    logger.debug(f"Starting single-pass scan for {len(failed_emails)} failed emails")

    # Single pass scan of the mbox
    for email in parse_mbox(mbox_path, None, config):
        if email.message_id in failed_ids_set:
            found_emails_dict[email.message_id] = email
            logger.debug(f"Found email with message_id={email.message_id}")
            # Stop if all emails found
            if len(found_emails_dict) == len(failed_ids_set):
                logger.debug("All failed emails found, stopping scan")
                break

    logger.console(f"Found {len(found_emails_dict)} of {len(failed_emails)} emails")

    # Process the found emails
    for failed_email in failed_emails:
        message_id = failed_email[0]
        subject = failed_email[1]
        previous_error = failed_email[2]

        processed += 1

        # Check for graceful shutdown request
        if shutdown_event.is_set():
            logger.console("\nShutdown requested. Completing current email and exiting...")
            logger.debug("Shutdown requested during retry processing")
            break

        subject_preview = subject[:25] if subject else "(no subject)"
        logger.console(f"Retrying email {processed}/{len(failed_emails)}: {subject_preview}...")
        logger.debug(f"Retrying email {processed}: message_id={message_id}, previous_error={previous_error}")

        # Retrieve email from dictionary instead of scanning again
        email = found_emails_dict.get(message_id)

        if email is None:
            logger.error(f"ERROR: Could not find email with message_id={message_id} in {mbox_path.name}")
            logger.debug(f"Result: error (email not found in mbox)")
            errors += 1
            continue

        if state_db.hash_exists(email.body_hash):
            skipped += 1
            state_db.remove_failed_email(mbox_path.name, message_id)
            logger.debug("Result: skipped (duplicate hash)")
            continue

        # Deduplication mechanism: uses SHA-256 hash of email body to detect duplicates.
        # Note: This approach considers emails with identical body content as duplicates,
        # even if subjects differ. This is intentional to handle forwarded messages or
        # multiple replies with quoted text. However, it may miss near-duplicates with
        # minor formatting changes.
        classification_prompt = config.llm.censor_prompt.format(
            subject=email.subject or "",
            from_address=email.from_address or "",
            date_iso=email.date_iso or "",
            body=email.body or ""
        )

        try:
            raw_response = llm_provider.get_response(classification_prompt)
        except Exception as e:
            errors += 1
            logger.debug(f"Result: error (classification failed: {e})")
            continue

        # Multi-language classification mapping: extracts first token from LLM response
        # and maps to standardized labels using keywords in Italian (lavoro/privata/mista),
        # English (work/private/mixed), Spanish (trabajo/privado/mixto), German
        # (arbeit/privat/gemischt), and French (travail/privée/mixte). This supports
        # multilingual LLM outputs while maintaining consistent internal label system.
        first_word = raw_response.strip().split()[0].lower() if raw_response.strip() else ""

        label = None
        if first_word in ("lavoro", "work", "trabajo", "arbeit", "travail"):
            label = "work"
        elif first_word in ("privata", "private", "privado", "privat", "privée"):
            label = "private"
        elif first_word in ("mista", "mixed", "mixto", "gemischt", "mixte"):
            label = "mixed"
        else:
            errors += 1
            logger.debug(f"Result: error (unknown classification: {first_word})")
            continue

        if label == "private":
            discarded += 1
            state_db.remove_failed_email(mbox_path.name, message_id)
            logger.debug("Result: discarded (private)")
            continue

        # Content extraction prompt selection: censor_clean_prompt is used for "work" emails
        # (already work-focused, minor cleanup needed), while censor_extract_prompt is used
        # for "mixed" emails (requires extraction of work-relevant portions, filtering out
        # personal/social content). Both prompts are defined in config and customized for
        # their specific extraction tasks.
        if label == "work":
            extract_prompt = config.llm.censor_clean_prompt
        else:
            extract_prompt = config.llm.censor_extract_prompt

        extraction_prompt = extract_prompt.format(
            subject=email.subject or "",
            from_address=email.from_address or "",
            body=email.body or ""
        )

        try:
            work_text = llm_provider.get_response(extraction_prompt)
        except Exception as e:
            errors += 1
            logger.debug(f"Result: error (extraction failed: {e})")
            continue

        if not work_text or not work_text.strip():
            discarded += 1
            state_db.remove_failed_email(mbox_path.name, message_id)
            logger.debug(f"Result: discarded ({label} with empty content)")
            continue

        try:
            chunks = chunk_text(work_text, config.indexing.chunk_size, config.indexing.chunk_overlap)
        except Exception as e:
            errors += 1
            logger.debug(f"Result: error (chunking failed: {e})")
            continue

        records = []
        try:
            for chunk in chunks:
                # Include subject in embedding text to strengthen semantic signal
                # Format: "Subject: {subject}\n\n{chunk_text}"
                # This ensures vector search can match queries against both subject and body
                text_with_subject = f"Subject: {email.subject}\n\n{chunk.text}"
                embedding = embedder.embed_passage(text_with_subject)
                chunk_id = f"{email.message_id}__{chunk.chunk_index}"
                metadata = {
                    "message_id": email.message_id,
                    "subject": email.subject,
                    "date_ts": email.date_ts,
                    "date_iso": email.date_iso,
                    "from_address": email.from_address,
                    "from_domain": email.from_domain,
                    "to_addresses": email.to_addresses,
                    "to_domains": email.to_domains,
                    "cc_addresses": email.cc_addresses,
                    "account": email.account,
                    "label": label,
                    "chunk_index": chunk.chunk_index,
                    "chunk_total": chunk.chunk_total,
                    "mbox_file": mbox_path.name,
                }
                records.append(ChunkRecord(
                    chunk_id=chunk_id,
                    text=text_with_subject,  # Include subject for consistency with embedding
                    embedding=embedding,
                    metadata=metadata,
                ))
        except Exception as e:
            errors += 1
            logger.debug(f"Result: error (embedding failed: {e})")
            continue

        try:
            vector_store.upsert(records)
        except Exception as e:
            errors += 1
            logger.debug(f"Result: error (vector store failed: {e})")
            continue

        state_db.add_hash(email.body_hash)
        state_db.remove_failed_email(mbox_path.name, message_id)
        indexed += 1
        logger.debug(f"Result: indexed ({label}, {len(chunks)} chunks)")

    logger.console(f"Completed retry: {mbox_path.name}")
    logger.console(f"  Processed: {processed} | Indexed: {indexed} | Skipped: {skipped} | Discarded: {discarded} | Errors: {errors}")
    logger.debug(f"Completed retry for: {mbox_path.name}, stats: processed={processed}, skipped={skipped}, discarded={discarded}, indexed={indexed}, errors={errors}")

    return {"processed": processed, "skipped": skipped, "discarded": discarded, "indexed": indexed, "errors": errors}

def signal_handler(signum, frame):
    """Handle shutdown signals (SIGINT, SIGTERM) gracefully.

    Sets the shutdown event to allow current email processing to complete
    before exiting.
    """
    shutdown_event.set()

def main() -> None:
    args = parse_args()
    logger = Logger(is_silent=args.silent, is_debug=args.debug, log_file_path=log_file)

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.debug("=" * 59)
    logger.debug(" " * 22 + "Session started" + " " * 22)
    logger.debug("=" * 59)
    logger.console("Starting Censor Agent")

    # Load configuration
    if Path(config_file).exists():
        config = load_config(config_file)
    else:
        logger.error(f"ERROR: Missing configuration file {config_file}")
        return

    # Validate configuration
    if not config_is_valid(config, logger):
        logger.debug("Configuration file is malformed, check console output")
        return

    # Validate mbox_paths when needed
    if not validate_mbox_paths(config, args, logger):
        logger.debug("mbox_paths validation failed, check console output")
        return

    # Initialize SQLite statedb
    logger.debug("State database initialized")
    state_db = StateDB(config.paths.state_db_path)
    
    # Initialize LLM provider
    llm_provider = None
    if config.llm.provider in ("ollama", "ollama cloud"):
        try:
            llm_provider = OllamaProvider(
                provider=config.llm.provider,
                local_url=getattr(config.llm, "ollama_local_url", ""),
                model=config.llm.ollama_model,
                api_key=getattr(config.llm, "ollama_api_key", None),
            )
            logger.debug(f"LLM provider initialized: {config.llm.provider} with model {config.llm.ollama_model}")
        except ConnectionError as e:
            logger.error(f"ERROR: Failed to initialize LLM provider: {e}")
            logger.debug(f"ERROR: Failed to initialize LLM provider: {e}")
            return
    elif config.llm.provider == "llama.cpp":
        try:
            llm_provider = LlamaCppProvider(
                model_path=config.llm.llamacpp_model_path,
            )
            logger.debug(f"LLM provider initialized: {config.llm.provider} with model {config.llm.llamacpp_model_path}")
        except ConnectionError as e:
            logger.error(f"ERROR: Failed to initialize LLM provider: {e}")
            logger.debug(f"ERROR: Failed to initialize LLM provider: {e}")
            return
    
    # Initialize embedder
    try:
        embedder = Embedder(
            model_name=config.indexing.embedding_model,
            cache_dir=config.paths.fastembed_cache_path,
            use_prefix=config.indexing.use_prefix,
        )
        logger.debug(f"Embedder initialized with model: {config.indexing.embedding_model}")
    except Exception as e:
        logger.error(f"ERROR: Failed to initialize embedder: {e}")
        logger.debug(f"ERROR: Failed to initialize embedder: {e}")
        return

    # Initialize vector store
    try:
        vector_store = VectorStore(
            db_path=Path(config.paths.vector_db_path),
            collection_name=config.indexing.collection_name,
        )
        logger.debug(f"Vector store initialized with collection: {config.indexing.collection_name}")
    except Exception as e:
        logger.error(f"ERROR: Failed to initialize vector store: {e}")
        logger.debug(f"ERROR: Failed to initialize vector store: {e}")
        return

    # Determine target mbox files to process
    if args.file:
        target_files = [args.file]
        logger.debug(f"Single file mode: processing {args.file}")
    elif args.retry:
        target_files = [args.retry]
        logger.debug(f"Retry mode: processing {args.retry}")
    else:
        target_files = config.paths.mbox_paths
        logger.debug(f"Standard mode: processing {len(target_files)} files from config")

    # Configuration header
    mode_label = "Single file" if args.file else "Retry" if args.retry else "Standard"
    llm_model = config.llm.ollama_model if config.llm.provider in ("ollama", "ollama cloud") else config.llm.llamacpp_model_path
    logger.console("")
    logger.console("=" * 50)
    logger.console("CONFIGURATION")
    logger.console("=" * 50)
    logger.console(f"Mode:             {mode_label}")
    if args.file:
        logger.console(f"Target file:      {args.file}")
    elif args.retry:
        logger.console(f"Retry file:       {args.retry}")
    logger.console(f"LLM Provider:     {config.llm.provider}")
    logger.console(f"LLM Model:        {llm_model}")
    logger.console(f"Embedder Model:   {config.indexing.embedding_model}")
    logger.console(f"Vector Store:     {config.paths.vector_db_path}")
    logger.console(f"Chunk Size:       {config.indexing.chunk_size}")
    logger.console("=" * 50)
    logger.console("")

    run_start_dt = datetime.now(timezone.utc)
    run_start = run_start_dt.isoformat()

    # Process mbox files
    total_processed = 0
    total_skipped = 0
    total_discarded = 0
    total_indexed = 0
    total_errors = 0

    if args.retry:
        # Retry mode: process failed emails for the single target file
        logger.console(f"Retry mode: processing {target_files[0]}")
        result = process_retries(Path(target_files[0]), config, state_db, llm_provider, embedder, vector_store, logger)
        total_processed = result["processed"]
        total_skipped = result["skipped"]
        total_discarded = result["discarded"]
        total_indexed = result["indexed"]
        total_errors = result["errors"]
    else:
        # Standard mode or single file mode: process all target files
        for mbox_idx, mbox_path in enumerate(target_files, start=1):
            logger.debug(f"Processing file {mbox_idx} of {len(target_files)}")
            logger.console(f"Processing file {mbox_idx} of {len(target_files)}")
            result = process_file(Path(mbox_path), config, state_db, llm_provider, embedder, vector_store, logger)
            total_processed += result["processed"]
            total_skipped += result["skipped"]
            total_discarded += result["discarded"]
            total_indexed += result["indexed"]
            total_errors += result["errors"]

    # Optimize vector store: compact data files and remove old versions
    logger.console("Optimizing vector store...")
    logger.debug("Starting vector store optimization (compact files + cleanup old versions)")
    vector_store.optimize()
    logger.debug("Vector store optimization completed")

    run_end_dt = datetime.now(timezone.utc)
    run_end = run_end_dt.isoformat()
    duration_seconds = (run_end_dt - run_start_dt).total_seconds()
    avg_time_per_email = duration_seconds / total_processed if total_processed > 0 else 0

    # Update state db_path
    state_db.log_run(run_start, run_end, total_processed, total_skipped, total_discarded, total_indexed, total_errors)

    # Check if shutdown was requested
    if shutdown_event.is_set():
        logger.console("\n*** PROCESS INTERRUPTED BY USER ***")
        logger.debug("Process interrupted by user signal")

    # Final summary
    logger.console("")
    logger.console("=" * 50)
    logger.console("SESSION SUMMARY")
    logger.console("=" * 50)
    logger.console(f"Total processed:  {total_processed}")
    logger.console(f"Total indexed:    {total_indexed}")
    logger.console(f"Total skipped:    {total_skipped}")
    logger.console(f"Total discarded:  {total_discarded}")
    logger.console(f"Total errors:     {total_errors}")
    logger.console(f"Duration:         {duration_seconds:.2f} seconds")
    logger.console(f"Avg time/email:   {avg_time_per_email:.2f} seconds")
    logger.console("=" * 50)

    # Goodbye
    logger.debug(f"Session ended in {duration_seconds:.2f} seconds, avg {avg_time_per_email:.2f}s/email")
    logger.console(f"Session ended in {duration_seconds:.2f} seconds")


if __name__ == "__main__":
    main()
