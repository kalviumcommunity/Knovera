"""
src/api/routes/guardrails.py

Persistent REST API for AI Safety Guardrails backed by Knovera MongoDB Atlas.
Supports full CRUD, priority reordering, active/inactive toggles, and per-admin isolation.
"""

import datetime
from typing import List, Optional
from fastapi import APIRouter, HTTPException, status, Query, Header
from pydantic import BaseModel

from src.services.mongo_storage import get_mongo_storage

router = APIRouter(prefix="/api/guardrails", tags=["Guardrails Management"])


class GuardrailModel(BaseModel):
    id: Optional[str] = None
    admin_id: Optional[str] = None
    name: str
    category: str
    description: Optional[str] = ""
    rule: str
    triggerCondition: Optional[str] = ""
    action: str
    priority: Optional[int] = 1
    enabled: bool = True
    createdAt: Optional[str] = None
    updatedAt: Optional[str] = None


class ReorderRequest(BaseModel):
    orderedIds: List[str]
    admin_id: Optional[str] = None


@router.get("", response_model=List[GuardrailModel])
def list_guardrails(
    admin_id: Optional[str] = Query(None, description="Filter guardrails by administrator identity"),
    x_admin_id: Optional[str] = Header(None, alias="X-Admin-Id")
):
    """Retrieve all guardrails for this administrator ordered by priority from MongoDB Atlas."""
    target_admin = admin_id or x_admin_id or "admin@knovera.ai"
    storage = get_mongo_storage()
    items = storage.list_guardrails(admin_id=target_admin)
    return [GuardrailModel(**item) for item in items]


@router.post("", response_model=GuardrailModel, status_code=status.HTTP_201_CREATED)
def create_guardrail(
    payload: GuardrailModel,
    admin_id: Optional[str] = Query(None),
    x_admin_id: Optional[str] = Header(None, alias="X-Admin-Id")
):
    """Insert a new safety guardrail into MongoDB Atlas."""
    target_admin = admin_id or x_admin_id or payload.admin_id or "admin@knovera.ai"
    storage = get_mongo_storage()
    data = payload.model_dump()
    data["admin_id"] = target_admin
    saved = storage.save_guardrail(data, admin_id=target_admin)
    return GuardrailModel(**saved)


@router.put("/{guardrail_id}", response_model=GuardrailModel)
def update_guardrail(
    guardrail_id: str,
    payload: GuardrailModel,
    admin_id: Optional[str] = Query(None),
    x_admin_id: Optional[str] = Header(None, alias="X-Admin-Id")
):
    """Update an existing guardrail in MongoDB Atlas."""
    target_admin = admin_id or x_admin_id or payload.admin_id or "admin@knovera.ai"
    storage = get_mongo_storage()
    data = payload.model_dump()
    data["id"] = guardrail_id
    data["admin_id"] = target_admin
    saved = storage.save_guardrail(data, admin_id=target_admin)
    return GuardrailModel(**saved)


@router.delete("/{guardrail_id}")
def delete_guardrail(
    guardrail_id: str,
    admin_id: Optional[str] = Query(None),
    x_admin_id: Optional[str] = Header(None, alias="X-Admin-Id")
):
    """Delete a guardrail from MongoDB Atlas."""
    target_admin = admin_id or x_admin_id
    storage = get_mongo_storage()
    storage.delete_guardrail(guardrail_id, admin_id=target_admin)
    return {"status": "deleted", "id": guardrail_id}


@router.post("/reorder")
def reorder_guardrails(
    payload: ReorderRequest,
    x_admin_id: Optional[str] = Header(None, alias="X-Admin-Id")
):
    """Update priority order for guardrails in MongoDB Atlas."""
    target_admin = payload.admin_id or x_admin_id or "admin@knovera.ai"
    storage = get_mongo_storage()
    for priority, gid in enumerate(payload.orderedIds, start=1):
        storage.save_guardrail({"id": gid, "priority": priority}, admin_id=target_admin)
    return {"status": "reordered", "count": len(payload.orderedIds)}
