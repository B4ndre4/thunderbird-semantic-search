import sqlite3
from pathlib import Path


class StateDB:
    SCHEMA = """
    CREATE TABLE IF NOT EXISTS state (
        mbox_file TEXT PRIMARY KEY,
        last_message_id TEXT
    );
    CREATE TABLE IF NOT EXISTS dedup (
        hash TEXT PRIMARY KEY
    );
    CREATE TABLE IF NOT EXISTS runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        start_time TEXT,
        end_time TEXT,
        processed INTEGER,
        skipped INTEGER,
        discarded INTEGER,
        indexed INTEGER,
        errors INTEGER
    );
    CREATE TABLE IF NOT EXISTS failed_emails (
        mbox_file TEXT NOT NULL,
        message_id TEXT NOT NULL,
        subject TEXT,
        error_message TEXT,
        last_failure TEXT,
        last_retry TEXT,
        retry_count INTEGER DEFAULT 0,
        PRIMARY KEY (mbox_file, message_id)
    );
    """

    def __init__(self, db_path: Path) -> None:
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(self.SCHEMA)

    def get_last_message_id(self, mbox_file: str) -> str | None:
        cursor = self._conn.execute(
            "SELECT last_message_id FROM state WHERE mbox_file = ?",
            (mbox_file,),
        )
        row = cursor.fetchone()
        return row[0] if row else None

    def set_last_message_id(self, mbox_file: str, message_id: str) -> None:
        self._conn.execute("BEGIN")
        self._conn.execute(
            "INSERT OR REPLACE INTO state (mbox_file, last_message_id) VALUES (?, ?)",
            (mbox_file, message_id),
        )
        self._conn.execute("COMMIT")

    def hash_exists(self, hash_hex: str) -> bool:
        cursor = self._conn.execute(
            "SELECT 1 FROM dedup WHERE hash = ?", (hash_hex,)
        )
        return cursor.fetchone() is not None

    def add_hash(self, hash_hex: str) -> None:
        self._conn.execute("BEGIN")
        self._conn.execute("INSERT OR IGNORE INTO dedup (hash) VALUES (?)", (hash_hex,))
        self._conn.execute("COMMIT")

    def log_run(self, start_time: str, end_time: str, processed: int,
                skipped: int, discarded: int, indexed: int, errors: int) -> None:
        self._conn.execute("BEGIN")
        self._conn.execute(
            "INSERT INTO runs (start_time, end_time, processed, skipped, discarded, indexed, errors) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (start_time, end_time, processed, skipped, discarded, indexed, errors),
        )
        self._conn.execute("COMMIT")

    def get_indexed_count(self) -> int:
        cursor = self._conn.execute("SELECT COUNT(*) FROM dedup")
        return cursor.fetchone()[0]

    def add_failed_email(self, mbox_file: str, message_id: str, subject: str, error: str) -> None:
        """Upsert a failed email with incremental retry counter.

        Uses INSERT OR REPLACE to atomically upsert a failed email record:
        - If new: creates record with retry_count=1
        - If existing: increments retry_count, updates error_message and last_failure

        last_failure always reflects the most recent failure timestamp, while
        last_retry tracks when the most recent retry attempt occurred.
        """
        self._conn.execute("BEGIN")
        self._conn.execute(
            """INSERT OR REPLACE INTO failed_emails 
               (mbox_file, message_id, subject, error_message, last_failure, last_retry, retry_count)
               VALUES (?, ?, ?, ?, datetime('now'), datetime('now'), 
                       COALESCE((SELECT retry_count FROM failed_emails WHERE mbox_file = ? AND message_id = ?), 0) + 1)
            """,
            (mbox_file, message_id, subject, error, mbox_file, message_id),
        )
        self._conn.execute("COMMIT")

    def get_failed_emails(self, mbox_file: str) -> list[tuple]:
        cursor = self._conn.execute(
            """SELECT message_id, subject, error_message, last_failure, last_retry, retry_count 
               FROM failed_emails WHERE mbox_file = ? ORDER BY last_failure""",
            (mbox_file,),
        )
        return cursor.fetchall()

    def remove_failed_email(self, mbox_file: str, message_id: str) -> None:
        self._conn.execute("BEGIN")
        self._conn.execute(
            "DELETE FROM failed_emails WHERE mbox_file = ? AND message_id = ?",
            (mbox_file, message_id),
        )
        self._conn.execute("COMMIT")

    def close(self) -> None:
        self._conn.commit()
        self._conn.close()
