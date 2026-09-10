"""
src/api/routes/conversations.py

Persistent Chat History & Conversation Storage backed by Knovera SQLite Database.
"""

import datetime
import json
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from src.db import get_db_connection

router = APIRouter(prefix="/api/conversations", tags=["Conversations History"])


class ChatTurnModel(BaseModel):
    id: str
    role: str
    content: str
    sources: Optional[List[Dict[str, Any]]] = None
    latency_ms: Optional[float] = None
    timestamp: Optional[str] = None
    feedback: Optional[str] = None


class ConversationModel(BaseModel):
    id: str
    title: str
    messages: List[ChatTurnModel] = []
    createdAt: str
    updatedAt: str


class CreateConversationRequest(BaseModel):
    id: Optional[str] = None
    title: Optional[str] = "New chat"


class RenameRequest(BaseModel):
    title: str


@router.get("", response_model=List[ConversationModel])
def list_conversations():
    """Retrieve all chat conversations with their message turns from SQLite database."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM conversations ORDER BY updated_at DESC")
    conv_rows = cursor.fetchall()

    conversations = []
    for c in conv_rows:
        cid = c["id"]
        cursor.execute("SELECT * FROM chat_messages WHERE conversation_id = ? ORDER BY rowid ASC", (cid,))
        msg_rows = cursor.fetchall()

        messages = []
        for m in msg_rows:
            sources = None
            if m["sources_json"]:
                try:
                    sources = json.loads(m["sources_json"])
                except Exception:
                    pass

            messages.append(
                ChatTurnModel(
                    id=m["id"],
                    role=m["role"],
                    content=m["content"],
                    sources=sources,
                    latency_ms=m["latency_ms"],
                    timestamp=m["timestamp"],
                    feedback=m["feedback"]
                )
            )

        conversations.append(
            ConversationModel(
                id=cid,
                title=c["title"],
                messages=messages,
                createdAt=c["created_at"],
                updatedAt=c["updated_at"]
            )
        )

    conn.close()
    return conversations


@router.post("", response_model=ConversationModel, status_code=status.HTTP_201_CREATED)
def create_conversation(payload: CreateConversationRequest):
    """Create a new conversation session in SQLite database."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cid = payload.id or f"conv_{int(datetime.datetime.now().timestamp() * 1000)}"
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()

    cursor.execute("""
        INSERT INTO conversations (id, title, created_at, updated_at)
        VALUES (?, ?, ?, ?)
    """, (cid, payload.title or "New chat", now, now))

    conn.commit()
    conn.close()

    return ConversationModel(
        id=cid,
        title=payload.title or "New chat",
        messages=[],
        createdAt=now,
        updatedAt=now
    )


@router.put("/{conversation_id}", response_model=ConversationModel)
def rename_conversation(conversation_id: str, payload: RenameRequest):
    """Rename a conversation in SQLite database."""
    conn = get_db_connection()
    cursor = conn.cursor()

    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    cursor.execute("""
        UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?
    """, (payload.title, now, conversation_id))

    if cursor.rowcount == 0:
        conn.close()
        raise HTTPException(status_code=404, detail="Conversation not found")

    cursor.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,))
    conv = cursor.fetchone()
    conn.commit()
    conn.close()

    return ConversationModel(
        id=conv["id"],
        title=conv["title"],
        messages=[],
        createdAt=conv["created_at"],
        updatedAt=conv["updated_at"]
    )


@router.delete("/{conversation_id}", status_code=status.HTTP_200_OK)
def delete_conversation(conversation_id: str):
    """Delete a conversation and all its messages from SQLite database."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM chat_messages WHERE conversation_id = ?", (conversation_id,))
    cursor.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
    conn.commit()
    conn.close()

    return {"status": "deleted", "id": conversation_id}


@router.post("/{conversation_id}/messages", response_model=ChatTurnModel)
def add_message(conversation_id: str, payload: ChatTurnModel):
    """Append a message turn to a conversation in SQLite database."""
    conn = get_db_connection()
    cursor = conn.cursor()

    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    sources_json = json.dumps(payload.sources) if payload.sources else None

    # Ensure conversation exists
    cursor.execute("SELECT id FROM conversations WHERE id = ?", (conversation_id,))
    if not cursor.fetchone():
        cursor.execute("""
            INSERT INTO conversations (id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?)
        """, (conversation_id, payload.content[:32] if payload.role == "user" else "New chat", now, now))

    cursor.execute("""
        INSERT INTO chat_messages (id, conversation_id, role, content, sources_json, latency_ms, timestamp, feedback)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        payload.id,
        conversation_id,
        payload.role,
        payload.content,
        sources_json,
        payload.latency_ms,
        payload.timestamp or now,
        payload.feedback
    ))

    cursor.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conversation_id))
    conn.commit()
    conn.close()

    return payload
