"""SQLite storage for conversation turns.

One row per `Turn` (see `app.factory.factory.Turn`) — same shape regardless
of which provider produced it, so a conversation can switch providers
mid-stream without a storage-format problem.
"""

import json
import sqlite3
from pathlib import Path

from app.factory.factory import Turn

DB_PATH = Path(__file__).resolve().parent.parent.parent / "chat.db"

_connection: sqlite3.Connection | None = None


def get_connection() -> sqlite3.Connection:
    global _connection
    if _connection is None:
        raise RuntimeError("db not initialized — call init_db() first")
    return _connection


def init_db() -> None:
    """Open the DB connection and create the schema if it doesn't exist. Call once at startup."""
    global _connection
    if _connection is not None:
        return
    _connection = sqlite3.connect(DB_PATH, check_same_thread=False)
    _connection.execute(
        """
        CREATE TABLE IF NOT EXISTS turns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            text TEXT,
            tool_calls TEXT,      -- JSON, only set on assistant turns that called tools
            tool_call_id TEXT,    -- only set on role="tool" turns
            tool_name TEXT,       -- only set on role="tool" turns
            result TEXT,          -- JSON, only set on role="tool" turns
            is_error INTEGER,     -- 0/1, only set on role="tool" turns
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    _connection.execute("CREATE INDEX IF NOT EXISTS idx_turns_session ON turns (session_id, id)")
    _connection.commit()


def save_turns(session_id: str, turns: list[Turn]) -> None:
    conn = get_connection()
    conn.executemany(
        """INSERT INTO turns
           (session_id, role, text, tool_calls, tool_call_id, tool_name, result, is_error)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            (
                session_id,
                t["role"],
                t.get("text"),
                json.dumps(t["tool_calls"]) if t.get("tool_calls") else None,
                t.get("tool_call_id"),
                t.get("tool_name"),
                json.dumps(t["result"]) if "result" in t else None,
                int(t["is_error"]) if "is_error" in t else None,
            )
            for t in turns
        ],
    )
    conn.commit()


def load_turns(session_id: str) -> list[Turn]:
    conn = get_connection()
    rows = conn.execute(
        """SELECT role, text, tool_calls, tool_call_id, tool_name, result, is_error
           FROM turns WHERE session_id = ? ORDER BY id""",
        (session_id,),
    ).fetchall()

    turns: list[Turn] = []
    for role, text, tool_calls, tool_call_id, tool_name, result, is_error in rows:
        turn: Turn = {"role": role}
        if text is not None:
            turn["text"] = text
        if tool_calls:
            turn["tool_calls"] = json.loads(tool_calls)
        if tool_call_id:
            turn["tool_call_id"] = tool_call_id
        if tool_name:
            turn["tool_name"] = tool_name
        if result is not None:
            turn["result"] = json.loads(result)
        if is_error is not None:
            turn["is_error"] = bool(is_error)
        turns.append(turn)
    return turns
