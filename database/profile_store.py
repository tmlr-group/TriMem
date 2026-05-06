"""
Profile Store - SQLite key-value store for entity profiles.

Stores progressive persona profiles built from extracted facts.
Each entity (person) gets a single text profile that is updated
incrementally as new facts are learned.
"""
import sqlite3
from typing import Optional, Dict


class ProfileStore:
    """
    Simple SQLite store for entity profiles, keyed by entity name.
    """

    def __init__(self, db_path: str = "./profile_store.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_table()

    def _init_table(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS profiles (
                entity_name TEXT PRIMARY KEY,
                profile_text TEXT NOT NULL
            )
        """)
        self.conn.commit()

    def get(self, entity_name: str) -> Optional[str]:
        """Fetch a single entity profile."""
        cursor = self.conn.execute(
            "SELECT profile_text FROM profiles WHERE entity_name = ?",
            (entity_name.lower(),)
        )
        row = cursor.fetchone()
        return row[0] if row else None

    def upsert(self, entity_name: str, profile_text: str):
        """Insert or update an entity profile."""
        self.conn.execute(
            """INSERT INTO profiles (entity_name, profile_text)
               VALUES (?, ?)
               ON CONFLICT(entity_name) DO UPDATE SET profile_text = excluded.profile_text""",
            (entity_name.lower(), profile_text)
        )
        self.conn.commit()

    def get_multiple(self, entity_names: list) -> Dict[str, str]:
        """Fetch profiles for multiple entities."""
        if not entity_names:
            return {}
        placeholders = ",".join("?" for _ in entity_names)
        cursor = self.conn.execute(
            f"SELECT entity_name, profile_text FROM profiles WHERE entity_name IN ({placeholders})",
            [name.lower() for name in entity_names]
        )
        return {row[0]: row[1] for row in cursor.fetchall()}

    def get_all(self) -> Dict[str, str]:
        """Fetch all profiles."""
        cursor = self.conn.execute("SELECT entity_name, profile_text FROM profiles")
        return {row[0]: row[1] for row in cursor.fetchall()}

    def clear(self):
        """Delete all profiles."""
        self.conn.execute("DELETE FROM profiles")
        self.conn.commit()

    def count(self) -> int:
        cursor = self.conn.execute("SELECT COUNT(*) FROM profiles")
        return cursor.fetchone()[0]
