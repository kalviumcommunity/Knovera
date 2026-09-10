"""
src/api/routes/guardrails.py

Persistent REST API for AI Safety Guardrails backed by Knovera SQLite Database.
Supports full CRUD, priority reordering, and active/inactive toggles.
"""

import datetime
from typing import List, Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from src.db import get_db_connection

router = APIRouter(prefix="/api/guardrails", tags=["Guardrails Management"])


class GuardrailModel(BaseModel):
    id: Optional[str] = None
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


@router.get("", response_model=List[GuardrailModel])
def list_guardrails():
    """Retrieve all guardrails ordered by priority ascending from SQLite database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM guardrails ORDER BY priority ASC")
    rows = cursor.fetchall()
    conn.close()

    result = []
    for r in rows:
        result.append(
            GuardrailModel(
                id=r["id"],
                name=r["name"],
                category=r["category"],
                description=r["description"] or "",
                rule=r["rule"],
                triggerCondition=r["trigger_condition"] or "",
                action=r["action"],
                priority=r["priority"],
                enabled=bool(r["enabled"]),
                createdAt=r["created_at"],
                updatedAt=r["updated_at"]
            )
        )
    return result


@router.post("", response_model=GuardrailModel, status_code=status.HTTP_201_CREATED)
def create_guardrail(payload: GuardrailModel):
    """Insert a new guardrail into SQLite database."""
    conn = get_db_connection()
    cursor = conn.cursor()

    gid = payload.id or f"gr_{int(datetime.datetime.now().timestamp() * 1000)}"
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # Determine priority if not set
    if not payload.priority:
        cursor.execute("SELECT COUNT(*) FROM guardrails")
        payload.priority = cursor.fetchone()[0] + 1

    cursor.execute("""
        INSERT INTO guardrails (id, name, category, description, rule, trigger_condition, action, priority, enabled, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        gid,
        payload.name,
        payload.category,
        payload.description or "",
        payload.rule,
        payload.triggerCondition or "",
        payload.action,
        payload.priority,
        1 if payload.enabled else 0,
        now,
        now
    ))
    conn.commit()
    conn.close()

    payload.id = gid
    payload.createdAt = now
    payload.updatedAt = now
    return payload


@router.put("/{guardrail_id}", response_model=GuardrailModel)
def update_guardrail(guardrail_id: str, payload: GuardrailModel):
    """Update an existing guardrail in SQLite database."""
    conn = get_db_connection()
    cursor = conn.cursor()

    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    cursor.execute("""
        UPDATE guardrails
        SET name = ?, category = ?, description = ?, rule = ?, trigger_condition = ?, action = ?, priority = ?, enabled = ?, updated_at = ?
        WHERE id = ?
    """, (
        payload.name,
        payload.category,
        payload.description or "",
        payload.rule,
        payload.triggerCondition or "",
        payload.action,
        payload.priority,
        1 if payload.enabled else 0,
        now,
        guardrail_id
    ))

    if cursor.rowcount == 0:
        conn.close()
        raise HTTPException(status_code=404, detail=f"Guardrail {guardrail_id} not found.")

    conn.commit()
    conn.close()

    payload.id = guardrail_id
    payload.updatedAt = now
    return payload


@router.delete("/{guardrail_id}", status_code=status.HTTP_200_OK)
def delete_guardrail(guardrail_id: str):
    """Delete a guardrail from SQLite database and renumber remaining priorities."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM guardrails WHERE id = ?", (guardrail_id,))
    if cursor.rowcount == 0:
        conn.close()
        raise HTTPException(status_code=404, detail=f"Guardrail {guardrail_id} not found.")

    # Renumber remaining priorities
    cursor.execute("SELECT id FROM guardrails ORDER BY priority ASC")
    remaining = cursor.fetchall()
    for idx, row in enumerate(remaining, start=1):
        cursor.execute("UPDATE guardrails SET priority = ? WHERE id = ?", (idx, row["id"]))

    conn.commit()
    conn.close()
    return {"status": "deleted", "id": guardrail_id}


@router.post("/reorder", status_code=status.HTTP_200_OK)
def reorder_guardrails(payload: ReorderRequest):
    """Reorder guardrail priorities based on ordered list of IDs."""
    conn = get_db_connection()
    cursor = conn.cursor()

    for idx, gid in enumerate(payload.orderedIds, start=1):
        cursor.execute("UPDATE guardrails SET priority = ? WHERE id = ?", (idx, gid))

    conn.commit()
    conn.close()
    return {"status": "reordered", "count": len(payload.orderedIds)}
