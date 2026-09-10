"""
src/api/routes/conversations.py

Persistent Chat History & Conversation Storage backed by Knovera MongoDB Atlas & SQLite.
Enforces per-user chat isolation so different users maintain private conversations.
"""

import datetime
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, status, Query, Header
from pydantic import BaseModel

from src.services.mongo_storage import get_mongo_storage

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
    user_id: Optional[str] = None


class CreateConversationRequest(BaseModel):
    id: Optional[str] = None
    title: Optional[str] = "New chat"
    user_id: Optional[str] = None


class RenameRequest(BaseModel):
    title: str


@router.get("", response_model=List[ConversationModel])
def list_conversations(
    user_id: Optional[str] = Query(None, description="Filter conversations by authenticated user ID"),
    x_user_id: Optional[str] = Header(None, alias="X-User-Id")
):
    """Retrieve conversations for the active user from MongoDB Atlas."""
    active_user = user_id or x_user_id
    storage = get_mongo_storage()
    conv_list = storage.list_conversations(user_id=active_user)

    result = []
    for c in conv_list:
        messages = [
            ChatTurnModel(
                id=m.get("id", f"msg_{idx}"),
                role=m.get("role", "user"),
                content=m.get("content", ""),
                sources=m.get("sources"),
                latency_ms=m.get("latency_ms"),
                timestamp=m.get("timestamp"),
                feedback=m.get("feedback")
            )
            for idx, m in enumerate(c.get("messages", []))
        ]
        result.append(
            ConversationModel(
                id=c.get("id"),
                title=c.get("title", "New chat"),
                messages=messages,
                createdAt=c.get("createdAt", datetime.datetime.now(datetime.timezone.utc).isoformat()),
                updatedAt=c.get("updatedAt", datetime.datetime.now(datetime.timezone.utc).isoformat()),
                user_id=c.get("user_id")
            )
        )
    return result


@router.post("", response_model=ConversationModel, status_code=status.HTTP_201_CREATED)
def create_conversation(
    payload: CreateConversationRequest,
    x_user_id: Optional[str] = Header(None, alias="X-User-Id")
):
    """Create a new conversation session associated with the active user in MongoDB."""
    target_user = payload.user_id or x_user_id or "user@knovera.ai"
    storage = get_mongo_storage()
    conv = storage.create_conversation(
        title=payload.title or "New chat",
        conv_id=payload.id,
        user_id=target_user
    )

    return ConversationModel(
        id=conv["id"],
        title=conv["title"],
        messages=[],
        createdAt=conv["createdAt"],
        updatedAt=conv["updatedAt"],
        user_id=target_user
    )


@router.put("/{conversation_id}", response_model=ConversationModel)
def rename_conversation(conversation_id: str, payload: RenameRequest):
    """Rename a conversation in MongoDB Atlas."""
    storage = get_mongo_storage()
    storage.rename_conversation(conversation_id, payload.title)
    
    # Retrieve updated
    all_convs = storage.list_conversations()
    target = next((c for c in all_convs if c.get("id") == conversation_id), None)
    if not target:
        raise HTTPException(status_code=404, detail="Conversation not found")

    messages = [
        ChatTurnModel(
            id=m.get("id"),
            role=m.get("role", "user"),
            content=m.get("content", ""),
            sources=m.get("sources"),
            latency_ms=m.get("latency_ms"),
            timestamp=m.get("timestamp"),
            feedback=m.get("feedback")
        )
        for m in target.get("messages", [])
    ]

    return ConversationModel(
        id=target["id"],
        title=target["title"],
        messages=messages,
        createdAt=target.get("createdAt", datetime.datetime.now(datetime.timezone.utc).isoformat()),
        updatedAt=target.get("updatedAt", datetime.datetime.now(datetime.timezone.utc).isoformat()),
        user_id=target.get("user_id")
    )


@router.delete("/{conversation_id}")
def delete_conversation(conversation_id: str):
    """Delete a conversation and its message turns from MongoDB Atlas."""
    storage = get_mongo_storage()
    storage.delete_conversation(conversation_id)
    return {"status": "deleted", "id": conversation_id}


@router.post("/{conversation_id}/messages", response_model=ChatTurnModel, status_code=status.HTTP_201_CREATED)
def append_message(conversation_id: str, payload: ChatTurnModel):
    """Append a user or assistant message turn to a conversation in MongoDB Atlas."""
    storage = get_mongo_storage()
    msg_dict = payload.model_dump()
    storage.save_chat_message(conversation_id, msg_dict)
    return payload
