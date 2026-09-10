"""
src/api/routes/logs.py

Persistent REST API for System & Chatbot Audit Logs backed by Knovera MongoDB Atlas.
Provides enterprise audit trail querying partitioned per administrator / tenant.
"""

import datetime
from typing import List, Optional
from fastapi import APIRouter, Query, Header, status
from pydantic import BaseModel

from src.services.mongo_storage import get_mongo_storage

router = APIRouter(prefix="/api/logs", tags=["Audit Logs"])


class LogModel(BaseModel):
    id: Optional[str] = None
    admin_id: Optional[str] = None
    timestamp: Optional[str] = None
    sessionId: Optional[str] = None
    user: str
    action: str
    querySnippet: Optional[str] = None
    latencyMs: float = 0.0
    groundednessScore: Optional[float] = None
    guardrailStatus: Optional[str] = None
    guardrailName: Optional[str] = None
    status: str
    details: Optional[str] = None


class LogListResponse(BaseModel):
    logs: List[LogModel]
    total: int
    page: int
    pageSize: int


@router.get("", response_model=LogListResponse)
def get_logs(
    search: Optional[str] = Query(None, description="Search keyword in action, user, or details"),
    status_filter: Optional[str] = Query(None, alias="status", description="Filter by status (success/warning/error)"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    admin_id: Optional[str] = Query(None),
    x_admin_id: Optional[str] = Header(None, alias="X-Admin-Id")
):
    """Retrieve audit logs from MongoDB Atlas with admin filtering and pagination."""
    target_admin = admin_id or x_admin_id or "admin@knovera.ai"
    storage = get_mongo_storage()
    data = storage.list_logs(
        admin_id=target_admin,
        search=search,
        status=status_filter,
        page=page,
        page_size=page_size
    )

    log_models = []
    for r in data.get("logs", []):
        log_models.append(
            LogModel(
                id=r.get("id"),
                admin_id=r.get("admin_id"),
                timestamp=r.get("timestamp"),
                sessionId=r.get("session_id") or r.get("sessionId") or "session",
                user=r.get("user", "user"),
                action=r.get("action", "action"),
                querySnippet=r.get("query_snippet") or r.get("querySnippet"),
                latencyMs=float(r.get("latency_ms") or r.get("latencyMs") or 0.0),
                groundednessScore=r.get("groundedness_score") or r.get("groundednessScore"),
                guardrailStatus=r.get("guardrail_status") or r.get("guardrailStatus"),
                guardrailName=r.get("guardrail_name") or r.get("guardrailName"),
                status=r.get("status", "success"),
                details=r.get("details")
            )
        )

    return LogListResponse(
        logs=log_models,
        total=data.get("total", len(log_models)),
        page=page,
        pageSize=page_size
    )


@router.post("", response_model=LogModel, status_code=status.HTTP_201_CREATED)
def record_log(
    payload: LogModel,
    admin_id: Optional[str] = Query(None),
    x_admin_id: Optional[str] = Header(None, alias="X-Admin-Id")
):
    """Record an audit log entry in MongoDB Atlas."""
    target_admin = admin_id or x_admin_id or payload.admin_id or "admin@knovera.ai"
    storage = get_mongo_storage()
    data = payload.model_dump()
    data["admin_id"] = target_admin
    data["session_id"] = payload.sessionId
    data["query_snippet"] = payload.querySnippet
    data["latency_ms"] = payload.latencyMs
    data["groundedness_score"] = payload.groundednessScore
    data["guardrail_status"] = payload.guardrailStatus
    data["guardrail_name"] = payload.guardrailName
    saved = storage.create_log(data, admin_id=target_admin)
    return payload
