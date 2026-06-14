"""
memory.py — In-memory conversation history store.
Each session_id maps to a list of chat messages.
Resets when the server restarts (perfect for demo/assignment use).
"""
from typing import List, Dict

# Global in-memory store: { session_id: [ {role, content}, ... ] }
_store: Dict[str, List[dict]] = {}


def get_history(session_id: str) -> List[dict]:
    """Return chat history for a session. Returns empty list if new session."""
    return _store.get(session_id, [])


def add_message(session_id: str, role: str, content: str) -> None:
    """Append a message to the session history."""
    if session_id not in _store:
        _store[session_id] = []
    _store[session_id].append({"role": role, "content": content})


def clear_history(session_id: str) -> None:
    """Clear conversation history for a session."""
    if session_id in _store:
        del _store[session_id]


def list_sessions() -> List[str]:
    """Return all active session IDs."""
    return list(_store.keys())
