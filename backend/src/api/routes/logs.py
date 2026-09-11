"""
src/api/routes/logs.py

Persistent REST API for System & Chatbot Audit Logs backed by Knovera MongoDB Atlas.
Provides enterprise audit trail querying partitioned per administrator / tenant.
"""

import datetime
from typing import List, Optional, Any
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
    input: Optional[str] = None
    output: Optional[str] = None
    sources: Optional[List[Any]] = None


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
    target_admin = admin_id if isinstance(admin_id, str) else (x_admin_id if isinstance(x_admin_id, str) else "admin@knovera.ai")
    effective_search = search if isinstance(search, str) else None
    effective_status = status_filter if isinstance(status_filter, str) else None
    effective_page = page if isinstance(page, int) else 1
    effective_page_size = page_size if isinstance(page_size, int) else 50

    storage = get_mongo_storage()
    data = storage.list_logs(
        admin_id=target_admin,
        search=effective_search,
        status=effective_status,
        page=effective_page,
        page_size=effective_page_size
    )

    # Prefetch conversations for session_ids to resolve exact AI outputs and sources
    logs_raw = data.get("logs", [])
    session_ids = list({
        (r.get("session_id") or r.get("sessionId"))
        for r in logs_raw
        if (r.get("session_id") or r.get("sessionId"))
    })

    conv_map = {}
    if session_ids and storage.is_connected:
        try:
            col_conv = storage._db["conversations"]
            cursor_convs = col_conv.find({"id": {"$in": session_ids}})
            for c in cursor_convs:
                conv_map[c.get("id")] = c.get("messages", [])
        except Exception as e:
            pass

    log_models = []
    for r in logs_raw:
        session_id = r.get("session_id") or r.get("sessionId") or "session"
        raw_input = r.get("input") or r.get("query_snippet") or r.get("querySnippet") or r.get("action")
        raw_output = r.get("output")
        raw_sources = r.get("sources")

        # Resolve exact output and sources from stored conversation if missing or placeholder
        is_placeholder_output = not raw_output or (isinstance(raw_output, str) and raw_output.startswith("Guardrail:"))
        is_missing_sources = not raw_sources or len(raw_sources) == 0

        if (is_placeholder_output or is_missing_sources) and session_id in conv_map:
            conv_msgs = conv_map[session_id]
            if conv_msgs:
                user_q = (raw_input or "").strip().lower()
                matching_asst = None
                # Search for corresponding assistant turn
                for i, m in enumerate(conv_msgs):
                    if m.get("role") == "user":
                        u_content = (m.get("content") or "").strip().lower()
                        if user_q and (user_q in u_content or u_content in user_q):
                            if i + 1 < len(conv_msgs) and conv_msgs[i + 1].get("role") == "assistant":
                                matching_asst = conv_msgs[i + 1]
                                break
                if not matching_asst:
                    # Pick the last assistant message in session
                    asst_turns = [m for m in conv_msgs if m.get("role") == "assistant"]
                    if asst_turns:
                        matching_asst = asst_turns[-1]

                if matching_asst:
                    if is_placeholder_output and matching_asst.get("content"):
                        raw_output = matching_asst.get("content")
                    if is_missing_sources and matching_asst.get("sources"):
                        raw_sources = matching_asst.get("sources")

                    # Backfill to MongoDB Atlas log document
                    if storage.is_connected and r.get("id"):
                        try:
                            storage._db["audit_logs"].update_one(
                                {"id": r.get("id")},
                                {"$set": {"output": raw_output, "sources": raw_sources}}
                            )
                        except Exception:
                            pass

        # Fallback extraction for seed / legacy logs if still not resolved
        if not raw_sources and r.get("details"):
            details_str = r.get("details", "")
            if "customer_sla_refund_terms" in details_str:
                raw_sources = [{"source": "customer_sla_refund_terms.docx", "score": r.get("groundedness_score") or 0.94, "doc_title": "Customer SLA & Refund Terms"}]
            elif "rag_api_specification_v2" in details_str:
                raw_sources = [{"source": "rag_api_specification_v2.md", "score": r.get("groundedness_score") or 0.98, "doc_title": "RAG API Specification v2"}]
            elif "enterprise_security_compliance" in details_str:
                raw_sources = [{"source": "enterprise_security_compliance_2026.pdf", "score": 1.0, "doc_title": "Enterprise Security Compliance"}]

        if not raw_output or (isinstance(raw_output, str) and raw_output.startswith("Guardrail:")):
            if r.get("guardrail_status") == "triggered" or r.get("status") in ["error", "refused"]:
                raw_output = r.get("details") or "Request blocked by Knovera Security Guardrails."
            elif r.get("details") and not r.get("details", "").startswith("Guardrail:"):
                raw_output = r.get("details")

        log_models.append(
            LogModel(
                id=r.get("id"),
                admin_id=r.get("admin_id"),
                timestamp=r.get("timestamp"),
                sessionId=session_id,
                user=r.get("user", "user"),
                action=r.get("action", "action"),
                querySnippet=r.get("query_snippet") or r.get("querySnippet"),
                latencyMs=float(r.get("latency_ms") or r.get("latencyMs") or 0.0),
                groundednessScore=r.get("groundedness_score") or r.get("groundednessScore"),
                guardrailStatus=r.get("guardrail_status") or r.get("guardrailStatus"),
                guardrailName=r.get("guardrail_name") or r.get("guardrailName"),
                status=r.get("status", "success"),
                details=r.get("details"),
                input=raw_input,
                output=raw_output,
                sources=raw_sources or []
            )
        )

    return LogListResponse(
        logs=log_models,
        total=data.get("total", len(log_models)),
        page=effective_page,
        pageSize=effective_page_size
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
    data["input"] = payload.input or payload.querySnippet
    data["output"] = payload.output or payload.details
    data["sources"] = payload.sources or []
    saved = storage.create_log(data, admin_id=target_admin)
    return payload
