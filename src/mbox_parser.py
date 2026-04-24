import email
import email.utils
import email.header
import hashlib
import mailbox
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from src.config import Config, EmailPatterns


@dataclass
class ParsedEmail:
    message_id: str
    subject: str
    date_ts: int
    date_iso: str
    from_address: str
    from_domain: str
    to_addresses: str
    to_domains: str
    cc_addresses: str
    account: str
    body: str
    body_hash: str


def _classify_email_type(
    msg: email.message.Message,
    body: str,
    patterns: EmailPatterns
) -> str:
    """Classifies email structure using configured regex patterns.

    Structural classification algorithm:

    Emails are classified into three categories to optimize embedding quality:
    1. 'forward' - Emails containing forwarded content (separate from main body)
    2. 'thread_reply' - Replies in email threads (extract only the new content)
    3. 'standalone' - Original emails with no quoted content (use full body)

    Classification order matters: forward detection takes precedence over thread
    replies, as forwarded emails may also contain reply patterns but should be
    handled differently.

    Args:
        msg: The email message object.
        body: The plain text body of the email.
        patterns: EmailPatterns instance containing regex patterns for classification.

    Returns:
        One of: 'forward', 'thread_reply', 'standalone'
    """
    subject = msg.get("Subject", "")

    # Check forward subject patterns
    for pattern in patterns.forward_subject:
        if re.match(pattern, subject, re.IGNORECASE):
            # Re:/R: indicates reply, not forward - skip these.
            # This check prevents misclassification because some email clients use
            # forward markers (Fwd:, I:, etc.) even in reply subjects when the
            # original subject already contained those patterns. We explicitly
            # exclude replies to ensure only actual forwards are classified as such.
            if re.match(r"^\s*(r|re)\s*:", subject, re.IGNORECASE):
                continue
            return "forward"

    # Check forward body patterns
    for pattern in patterns.forward_body:
        if re.search(pattern, body, re.IGNORECASE | re.MULTILINE):
            return "forward"

    # Check thread reply patterns
    for pattern in patterns.thread_reply:
        if re.search(pattern, body, re.MULTILINE | re.IGNORECASE):
            return "thread_reply"

    return "standalone"


def _extract_plain_body(msg: email.message.Message) -> str:
    """Extract plain text body from email with charset fallback strategy.

    Strategy:
    1. For multipart emails: walk through all parts and find the first text/plain
    2. Use the declared charset from Content-Type header, fallback to utf-8
    3. If decoding fails with declared charset, try utf-8 as fallback
    4. Use 'replace' error handling to avoid data loss on malformed sequences

    This handles edge cases like:
    - Emails with incorrect charset declarations
    - Mixed encodings within the same mbox
    - Missing or malformed Content-Type headers
    """
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                charset = part.get_content_charset() or "utf-8"
                try:
                    payload = part.get_payload(decode=True)
                    if payload is None:
                        return ""
                    return payload.decode(charset, errors="replace")
                except (LookupError, UnicodeDecodeError):
                    try:
                        return payload.decode("utf-8", errors="replace")
                    except Exception:
                        return ""
        return ""
    else:
        charset = msg.get_content_charset() or "utf-8"
        try:
            payload = msg.get_payload(decode=True)
            if payload is None:
                return ""
            return payload.decode(charset, errors="replace")
        except (LookupError, UnicodeDecodeError):
            try:
                return payload.decode("utf-8", errors="replace")
            except Exception:
                return ""


def _extract_clean_thread_body(body: str, patterns: EmailPatterns) -> str:
    """Extract clean body from thread reply by removing previous conversation text.

    Algorithm:
    1. Scan the body for known thread reply markers (e.g., "On [date] [name] wrote:")
    2. If found, return only the text before the first marker
    3. If no marker found, return the full body (might be a partial quote)

    This preserves the actual new content while discarding quoted history,
    which is crucial for semantic search to avoid duplicate/repeated content
    in embeddings.

    Args:
        body: The original email body.
        patterns: EmailPatterns instance containing thread_reply patterns.

    Returns:
        The body text before the first thread reply marker, or the full body
        if no marker found.
    """
    for pattern in patterns.thread_reply:
        match = re.search(pattern, body, re.MULTILINE | re.IGNORECASE)
        if match:
            return body[:match.start()].strip()
    return body


def _extract_account(msg: email.message.Message) -> str:
    """Extract the recipient account using header priority hierarchy.

    Header priority hierarchy (most to least reliable):
    1. Delivered-To - The final delivery address after all forwarding/aliasing
    2. X-Original-To - The original recipient before any forwarding rules
    3. To - The address in the To: header (may be a list or alias)

    This hierarchy ensures we capture the actual account that received the email,
    not just the address listed in the visible headers which may be a mailing list
    or forwarded address.
    """
    for header in ("Delivered-To", "X-Original-To", "To"):
        value = msg.get(header, "")
        if value:
            address = email.utils.parseaddr(value)
            if address[1]:
                return address[1].lower()
    return "unknown"


def _decode_header_value(raw: str) -> str:
    parts = email.header.decode_header(raw)
    result = []
    for fragment, charset in parts:
        if isinstance(fragment, bytes):
            charset = charset or "utf-8"
            try:
                result.append(fragment.decode(charset, errors="replace"))
            except (LookupError, UnicodeDecodeError):
                result.append(fragment.decode("utf-8", errors="replace"))
        else:
            result.append(fragment)
    return "".join(result)


def parse_mbox(
    mbox_path: Path,
    skip_before_id: str | None,
    config: Config
) -> Iterator[ParsedEmail]:
    """Parse emails from an mbox file.

    Args:
        mbox_path: Path to the mbox file.
        skip_before_id: Message ID to skip until (for resuming). None to start from beginning.
        config: Application configuration containing email patterns.

    Yields:
        ParsedEmail objects for each email in the mbox.
    """
    try:
        mbox = mailbox.mbox(str(mbox_path), factory=None, create=False)
    except Exception:
        return

    # Resume logic: skip_before_id enables checkpointing for incremental processing.
    #
    # Checkpointing strategy:
    # - If skip_before_id is None: start from the beginning (no checkpoint)
    # - If skip_before_id is provided: skip all emails until we find it, then
    #   start processing from the next email
    #
    # This allows the parser to resume interrupted processing without starting
    # over, using message_id as a stable identifier. The flag past_checkpoint
    # tracks whether we've reached the resume point.
    past_checkpoint = skip_before_id is None

    for index, msg in enumerate(mbox):
        try:
            message_id_raw = msg.get("Message-ID", "").strip()
            if not message_id_raw:
                # Generate synthetic ID for emails without Message-ID header
                # This ensures every email can be checkpointed, even malformed ones
                message_id = f"nomsgid_{mbox_path.name}_{index}"
            else:
                message_id = message_id_raw

            # Checkpoint evaluation: skip until we pass the resume marker
            if not past_checkpoint:
                if message_id == skip_before_id:
                    past_checkpoint = True
                continue

            date_iso = "1970-01-01T00:00:00Z"
            date_ts = 0
            date_raw = msg.get("Date", "")
            if date_raw:
                try:
                    dt = email.utils.parsedate_to_datetime(date_raw)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    date_ts = int(dt.timestamp())
                    date_iso = dt.isoformat()
                except Exception:
                    pass

            from_raw = msg.get("From", "")
            from_parsed = email.utils.parseaddr(from_raw)
            from_address = from_parsed[1].lower() if from_parsed[1] else ""
            from_domain = ""
            if "@" in from_address:
                from_domain = from_address.split("@", 1)[1].lower()

            to_raw = msg.get("To", "")
            to_addrs = email.utils.getaddresses([to_raw] if to_raw else [])
            to_addresses = ", ".join(addr.lower() for _, addr in to_addrs if addr)
            to_domains = ", ".join(
                addr.split("@", 1)[1].lower()
                for _, addr in to_addrs
                if addr and "@" in addr
            )

            cc_raw = msg.get("Cc", "")
            cc_addrs = email.utils.getaddresses([cc_raw] if cc_raw else [])
            cc_addresses = ", ".join(addr.lower() for _, addr in cc_addrs if addr)

            subject_raw = msg.get("Subject", "")
            subject = _decode_header_value(subject_raw)

            body = _extract_plain_body(msg)

            email_type = _classify_email_type(msg, body, config.email_patterns)

            if email_type == "thread_reply":
                body = _extract_clean_thread_body(body, config.email_patterns)

            body_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()

            account = _extract_account(msg)

            yield ParsedEmail(
                message_id=message_id,
                subject=subject,
                date_ts=date_ts,
                date_iso=date_iso,
                from_address=from_address,
                from_domain=from_domain,
                to_addresses=to_addresses,
                to_domains=to_domains,
                cc_addresses=cc_addresses,
                account=account,
                body=body,
                body_hash=body_hash,
            )

        except Exception:
            continue


def get_email_by_message_id(
    mbox_path: Path,
    target_message_id: str,
    config: Config
) -> ParsedEmail | None:
    """Extract a specific email from mbox by message_id.

    Args:
        mbox_path: Path to the mbox file.
        target_message_id: Message ID to search for.
        config: Application configuration containing email patterns.

    Returns:
        ParsedEmail if found, None otherwise.
    """
    try:
        mbox = mailbox.mbox(str(mbox_path), factory=None, create=False)
    except Exception:
        return None

    for index, msg in enumerate(mbox):
        try:
            message_id_raw = msg.get("Message-ID", "").strip()
            if not message_id_raw:
                message_id = f"nomsgid_{mbox_path.name}_{index}"
            else:
                message_id = message_id_raw

            if message_id != target_message_id:
                continue

            date_iso = "1970-01-01T00:00:00Z"
            date_ts = 0
            date_raw = msg.get("Date", "")
            if date_raw:
                try:
                    dt = email.utils.parsedate_to_datetime(date_raw)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    date_ts = int(dt.timestamp())
                    date_iso = dt.isoformat()
                except Exception:
                    pass

            from_raw = msg.get("From", "")
            from_parsed = email.utils.parseaddr(from_raw)
            from_address = from_parsed[1].lower() if from_parsed[1] else ""
            from_domain = ""
            if "@" in from_address:
                from_domain = from_address.split("@", 1)[1].lower()

            to_raw = msg.get("To", "")
            to_addrs = email.utils.getaddresses([to_raw] if to_raw else [])
            to_addresses = ", ".join(addr.lower() for _, addr in to_addrs if addr)
            to_domains = ", ".join(
                addr.split("@", 1)[1].lower()
                for _, addr in to_addrs
                if addr and "@" in addr
            )

            cc_raw = msg.get("Cc", "")
            cc_addrs = email.utils.getaddresses([cc_raw] if cc_raw else [])
            cc_addresses = ", ".join(addr.lower() for _, addr in cc_addrs if addr)

            subject_raw = msg.get("Subject", "")
            subject = _decode_header_value(subject_raw)

            body = _extract_plain_body(msg)

            email_type = _classify_email_type(msg, body, config.email_patterns)

            if email_type == "thread_reply":
                body = _extract_clean_thread_body(body, config.email_patterns)

            body_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()

            account = _extract_account(msg)

            return ParsedEmail(
                message_id=message_id,
                subject=subject,
                date_ts=date_ts,
                date_iso=date_iso,
                from_address=from_address,
                from_domain=from_domain,
                to_addresses=to_addresses,
                to_domains=to_domains,
                cc_addresses=cc_addresses,
                account=account,
                body=body,
                body_hash=body_hash,
            )

        except Exception:
            continue

    return None


def count_remaining_emails(mbox_path: Path, skip_before_id: str | None) -> int:
    """Count emails remaining to process in mbox file.

    If skip_before_id is provided, count only emails AFTER that message_id.
    This mirrors the logic in parse_mbox() to ensure consistency.
    """
    try:
        mbox = mailbox.mbox(str(mbox_path), factory=None, create=False)
    except Exception:
        return 0

    past_checkpoint = skip_before_id is None
    count = 0

    for index, msg in enumerate(mbox):
        try:
            message_id_raw = msg.get("Message-ID", "").strip()
            if not message_id_raw:
                message_id = f"nomsgid_{mbox_path.name}_{index}"
            else:
                message_id = message_id_raw

            if not past_checkpoint:
                if message_id == skip_before_id:
                    past_checkpoint = True
                continue

            count += 1
        except Exception:
            continue

    return count
