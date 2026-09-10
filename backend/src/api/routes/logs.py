"""
src/api/routes/logs.py

Persistent REST API for System & Chatbot Audit Logs backed by Knovera SQLite Database.
"""

import datetime
from typing import List, Optional
from fastapi import APIRouter, Query, status
from pydantic import BaseModel

from src.db import get_db_connection

router = APIRouter(prefix="/api/logs", tags=["Audit Logs"])


class LogModel(BaseModel):
    id: Optional[str] = None
    timestamp: Optional[str] = None
    sessionId: str
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
    search: Optional[str] = Query(None, description="Search keyword in action, user, or session"),
    status_filter: Optional[str] = Query(None, alias="status", description="Filter by status (success/warning/error)"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200)
):
    """Retrieve audit logs from SQLite database with filtering and pagination."""
    conn = get_db_connection()
    cursor = conn.cursor()

    query_parts = ["SELECT * FROM audit_logs WHERE 1=1"]
    count_parts = ["SELECT COUNT(*) FROM audit_logs WHERE 1=1"]
    params = []

    if status_filter and status_filter != "all":
        query_parts.append("AND status = ?")
        count_parts.append("AND status = ?")
        params.append(status_filter)

    if search:
        search_like = f"%{search}%"
        condition = "AND (action LIKE ? OR user LIKE ? OR session_id LIKE ? OR query_snippet LIKE ?)"
        query_parts.append(condition)
        count_parts.append(condition)
        params.extend([search_like, search_like, search_like, search_like])

    cursor.execute(" ".join(count_parts), params)
    total = cursor.fetchone()[0]

    offset = (page - 1) * page_size
    query_parts.append("ORDER BY timestamp DESC LIMIT ? OFFSET ?")
    fetch_params = params + [page_size, offset]

    cursor.execute(" ".join(query_parts), fetch_params)
    rows = cursor.fetchall()
    conn.close()

    logs = []
    for r in rows:
        logs.append(
            LogModel(
                id=r["id"],
                timestamp=r["timestamp"],
                sessionId=r["session_id"],
                user=r["user"],
                action=r["action"],
                querySnippet=r["query_snippet"],
                latencyMs=r["latency_ms"],
                groundednessScore=r["groundedness_score"],
                guardrailStatus=r["guardrail_status"],
                guardrailName=r["guardrail_name"],
                status=r["status"],
                details=r["details"]
            )
        )

    return LogListResponse(
        logs=logs,
        total=total,
        page=page,
        pageSize=page_size
    )


@router.post("", response_model=LogModel, status_code=status.HTTP_201_CREATED)
def record_log(payload: LogModel):
    """Insert a new audit event into SQLite database."""
    conn = get_db_connection()
    cursor = conn.cursor()

    lid = payload.id or f"log_{int(datetime.datetime.now().timestamp() * 1000)}"
    ts = payload.timestamp or datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute("""
        INSERT INTO audit_logs (id, timestamp, session_id, user, action, query_snippet, latency_ms, groundedness_score, guardrail_status, guardrail_name, status, details)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        lid,
        ts,
        payload.sessionId,
        payload.user,
        payload.action,
        payload.querySnippet,
        payload.latencyMs,
        payload.groundednessScore,
        payload.guardrailStatus,
        payload.guardrailName,
        payload.status,
        payload.details
    ))
    conn.commit()
    conn.close()

    payload.id = lid
    payload.timestamp = ts
    return payload
