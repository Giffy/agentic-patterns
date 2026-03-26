import sqlite3
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class SQLiteShortTermMemory:
    """
    Implements a short-term memory system (Memory Management Pattern) 
    using SQLite for local persistence.
    Useful for retaining session context across agent steps or chat iterations.
    """
    
    def __init__(self, db_path: str = "short_term_memory.db"):
        """
        Initializes the SQLite database and creates the memory table if it doesn't exist.
        
        Args:
            db_path: Path to the SQLite database file. Defaults to 'short_term_memory.db'.
                     Use ':memory:' for truly ephemeral RAM-only storage.
        """
        self.db_path = db_path
        self._init_db()
        
    def _get_connection(self):
        return sqlite3.connect(self.db_path)
        
    def _init_db(self):
        """Creates the necessary schema."""
        query = '''
            CREATE TABLE IF NOT EXISTS memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        '''
        with self._get_connection() as conn:
            conn.execute(query)
            # Create an index for faster retrieval by session
            conn.execute('CREATE INDEX IF NOT EXISTS idx_session_id ON memory(session_id)')
            conn.commit()
        logger.info(f"[SQLiteMemory] Initialized short-term memory at {self.db_path}")

    def add_memory(self, session_id: str, role: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Stores a new interaction in the memory.
        
        Args:
            session_id: The unique identifier for the current conversation or workflow run.
            role: The role of the speaker (e.g., 'user', 'agent', 'system', 'tool').
            content: The text content to store.
            metadata: Optional dictionary of additional context (e.g., tool names, tokens).
        """
        meta_str = json.dumps(metadata) if metadata else None
        
        query = '''
            INSERT INTO memory (session_id, role, content, metadata)
            VALUES (?, ?, ?, ?)
        '''
        with self._get_connection() as conn:
            conn.execute(query, (session_id, role, content, meta_str))
            conn.commit()
            
    def get_context(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieves the most recent context/messages for a session.
        
        Args:
            session_id: The session to retrieve.
            limit: Maximum number of recent messages to return.
            
        Returns:
            A list of message dictionaries ordered chronologically.
        """
        query = '''
            SELECT role, content, metadata, timestamp 
            FROM memory 
            WHERE session_id = ? 
            ORDER BY timestamp DESC, id DESC 
            LIMIT ?
        '''
        results = []
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, (session_id, limit))
            rows = cursor.fetchall()
            
            # Since we ordered DESC to get the latest `limit` rows, we must reverse them 
            # to return them in regular chronological order (oldest -> newest in the window).
            for row in reversed(rows):
                results.append({
                    "role": row["role"],
                    "content": row["content"],
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else None,
                    "timestamp": row["timestamp"]
                })
        return results

    def clear_session(self, session_id: str):
        """
        Deletes all short-term memory associated with a specific session.
        """
        query = 'DELETE FROM memory WHERE session_id = ?'
        with self._get_connection() as conn:
            cursor = conn.execute(query, (session_id,))
            deleted = cursor.rowcount
            conn.commit()
        logger.info(f"[SQLiteMemory] Cleared {deleted} messages for session {session_id}")
        
    def format_as_string(self, session_id: str, limit: int = 10) -> str:
        """
        Utility method to retrieve memory and format it securely as a string block
        for easy LLM context injection.
        """
        history = self.get_context(session_id, limit)
        if not history:
            return "No previous context."
            
        formatted = []
        for msg in history:
            role = msg["role"].upper()
            formatted.append(f"{role}: {msg['content']}")
            
        return "\n\n".join(formatted)

    def get_exact_match_answer(self, query: str) -> Optional[str]:
        """
        Looks through the global memory for an exact match of a 'user' query
        and retrieves the very next 'assistant' response. This serves as a 
        simple, robust caching layer to avoid redundant LLM calls.
        """
        # We join the memory on itself where m2 is directly after m1 (the user query)
        query_sql = '''
            SELECT m2.content 
            FROM memory m1
            JOIN memory m2 ON m1.session_id = m2.session_id AND m2.id > m1.id
            WHERE m1.role = 'user' AND m1.content = ? AND m2.role = 'assistant'
            ORDER BY m2.id ASC LIMIT 1
        '''
        with self._get_connection() as conn:
            cursor = conn.execute(query_sql, (query,))
            row = cursor.fetchone()
            if row:
                return row[0]
        return None
