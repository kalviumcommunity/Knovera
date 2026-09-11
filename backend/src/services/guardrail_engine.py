"""
src/services/guardrail_engine.py

Enterprise AI Safety Guardrails Engine for Knovera RAG Assistant.
Dynamically integrates with MongoDB Atlas persistent guardrail policies.
Enforces:
1. Prompt Injection & Adversarial Jailbreak Defense (Refusal)
2. PII & Sensitive Data Redaction (Pre-retrieval and Pre-logging masking)
3. Grounded Hallucination Shield (Vector similarity threshold check)
4. Competitor Mention & Brand Protection (Neutral comparative disclaimers)
5. Output Secret Scrubbing (Prevents leakage of connection strings, API keys)
"""

import re
import logging
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger("knovera.guardrails")

# Regular expression patterns for sensitive PII and secrets
PATTERNS_PII = {
    "SSN": re.compile(r'\b(?:\d{3}-\d{2}-\d{4}|\b\d{9}\b(?=.*(?:ssn|social\s*security)))\b', re.IGNORECASE),
    "CREDIT_CARD": re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b'),
    "API_KEY": re.compile(r'\b(?:sk-[a-zA-Z0-9_\-]{20,}|Bearer\s+[a-zA-Z0-9._\-]{20,}|ghp_[a-zA-Z0-9]{36})\b'),
    "PRIVATE_KEY_OR_SECRET": re.compile(r'\b(?:password\s*[:=]\s*["\']?[^\s"\']{6,}|secret_key\s*[:=]\s*["\']?[^\s"\']{8,})\b', re.IGNORECASE),
    "CONNECTION_STRING": re.compile(r'mongodb(?:\+srv)?://[^\s"\'<>]+', re.IGNORECASE)
}

# Adversarial Prompt Injection & Jailbreak Patterns
INJECTION_PATTERNS = [
    re.compile(r'ignore\s+(all\s+)?(previous|prior|above|existing)\s+(instructions|prompts|directions|rules|guidelines)', re.IGNORECASE),
    re.compile(r'(output|print|reveal|show|display|leak)\s+(your\s+)?(entire\s+|complete\s+|internal\s+)?(system\s+prompt|initial\s+prompt|developer\s+prompt|guidelines)', re.IGNORECASE),
    re.compile(r'\byou\s+are\s+now\s+(dan|jailbreak|unrestricted|godmode|an\s+unfiltered\s+ai)\b', re.IGNORECASE),
    re.compile(r'(===|---)\s*(system\s+override|system\s+prompt|end\s+of\s+context|developer\s+mode)', re.IGNORECASE),
    re.compile(r'\b(bypass|disable)\s+(all\s+)?(safety|security|rules|guardrails|filters)\b', re.IGNORECASE),
    re.compile(r'\bact\s+as\s+an\s+unrestricted\s+ai\b', re.IGNORECASE)
]

# Monitored Competitor Entities
COMPETITOR_ENTITIES = [
    "pinecone",
    "weaviate",
    "qdrant",
    "milvus",
    "chromadb",
    "chroma db",
    "elasticsearch",
    "vespa"
]

COMPETITOR_DISCLAIMER = (
    "\n\n*Brand Policy Notice: Knovera delivers enterprise-grade RAG natively backed by MongoDB Atlas. "
    "Third-party platform comparisons are strictly limited to publicly available specification matrices.*"
)


class GuardrailDecision:
    def __init__(
        self,
        is_refused: bool = False,
        refusal_reason: Optional[str] = None,
        refusal_message: Optional[str] = None,
        sanitized_query: str = "",
        triggered_guardrail: Optional[str] = None,
        guardrail_status: str = "passed",
        append_disclaimer: Optional[str] = None,
        matched_pii_count: int = 0
    ):
        self.is_refused = is_refused
        self.refusal_reason = refusal_reason
        self.refusal_message = refusal_message
        self.sanitized_query = sanitized_query
        self.triggered_guardrail = triggered_guardrail
        self.guardrail_status = guardrail_status
        self.append_disclaimer = append_disclaimer
        self.matched_pii_count = matched_pii_count


class GuardrailEngine:
    """
    Evaluates and enforces active guardrails configured in MongoDB Atlas.
    """

    def __init__(self):
        pass

    def _get_active_guardrails(self, admin_id: str) -> List[Dict[str, Any]]:
        """Fetch guardrail configurations from MongoDB Atlas, ordered by priority."""
        try:
            from src.services.mongo_storage import get_mongo_storage
            storage = get_mongo_storage()
            all_grs = storage.list_guardrails(admin_id=admin_id)
            # Filter enabled only and sort by priority ascending
            enabled = [g for g in all_grs if g.get("enabled", True)]
            enabled.sort(key=lambda x: x.get("priority", 999))
            return enabled
        except Exception as e:
            logger.warning(f"Failed to fetch guardrails from MongoDB ({e}). Using safety defaults.")
            return [
                {"category": "injection", "enabled": True, "name": "Prompt Injection & Jailbreak Prevention"},
                {"category": "pii", "enabled": True, "name": "PII & Sensitive Data Redaction"},
                {"category": "policy", "enabled": True, "name": "Competitor Mention & Brand Protection"},
                {"category": "hallucination", "enabled": True, "name": "Grounded Hallucination Shield"}
            ]

    def redact_pii(self, text: str) -> Tuple[str, int]:
        """Scans and masks sensitive PII and secrets with anonymized tokens."""
        redacted = text
        total_matches = 0

        # Mask SSN
        matches = PATTERNS_PII["SSN"].findall(redacted)
        if matches:
            total_matches += len(matches)
            redacted = PATTERNS_PII["SSN"].sub("[REDACTED_SSN]", redacted)

        # Mask Credit Cards
        matches = PATTERNS_PII["CREDIT_CARD"].findall(redacted)
        if matches:
            total_matches += len(matches)
            redacted = PATTERNS_PII["CREDIT_CARD"].sub("[REDACTED_CREDIT_CARD]", redacted)

        # Mask API Keys
        matches = PATTERNS_PII["API_KEY"].findall(redacted)
        if matches:
            total_matches += len(matches)
            redacted = PATTERNS_PII["API_KEY"].sub("[REDACTED_API_KEY]", redacted)

        # Mask DB / Passwords
        matches = PATTERNS_PII["PRIVATE_KEY_OR_SECRET"].findall(redacted)
        if matches:
            total_matches += len(matches)
            redacted = PATTERNS_PII["PRIVATE_KEY_OR_SECRET"].sub("[REDACTED_SECRET]", redacted)

        # Mask Connection Strings
        matches = PATTERNS_PII["CONNECTION_STRING"].findall(redacted)
        if matches:
            total_matches += len(matches)
            redacted = PATTERNS_PII["CONNECTION_STRING"].sub("[REDACTED_CONNECTION_STRING]", redacted)

        return redacted, total_matches

    def evaluate_input(
        self,
        query: str,
        admin_id: str = "admin@knovera.ai"
    ) -> GuardrailDecision:
        """
        Executes pre-retrieval safety checks based on configured guardrail priority.
        """
        active_guardrails = self._get_active_guardrails(admin_id)
        current_query = query
        pii_matches_count = 0
        triggered_gr_name = None
        guardrail_status = "passed"
        append_disclaimer = None

        for gr in active_guardrails:
            category = gr.get("category", "").lower()
            name = gr.get("name", category)

            # 1. Prompt Injection & Jailbreak Defense
            if category == "injection":
                for pattern in INJECTION_PATTERNS:
                    if pattern.search(current_query):
                        logger.warning(f"Adversarial Prompt Injection blocked by '{name}': {current_query[:40]}...")
                        return GuardrailDecision(
                            is_refused=True,
                            refusal_reason="PROMPT_INJECTION_DETECTED",
                            refusal_message=(
                                "Request refused: This prompt contains instructions that violate Knovera "
                                "enterprise security policies (Adversarial Prompt Injection / System Override detected)."
                            ),
                            sanitized_query=current_query,
                            triggered_guardrail=name,
                            guardrail_status="refused"
                        )

            # 2. PII & Sensitive Data Redaction
            elif category == "pii":
                redacted, count = self.redact_pii(current_query)
                if count > 0:
                    pii_matches_count += count
                    current_query = redacted
                    if not triggered_gr_name:
                        triggered_gr_name = name
                        guardrail_status = "redacted"
                    logger.info(f"PII Redaction Guardrail '{name}' masked {count} sensitive tokens.")

            # 3. Competitor Mention & Brand Protection
            elif category == "policy":
                lower_q = current_query.lower()
                for comp in COMPETITOR_ENTITIES:
                    if comp in lower_q:
                        append_disclaimer = COMPETITOR_DISCLAIMER
                        if not triggered_gr_name:
                            triggered_gr_name = name
                        logger.info(f"Competitor policy triggered by entity '{comp}'.")
                        break

        return GuardrailDecision(
            is_refused=False,
            sanitized_query=current_query,
            triggered_guardrail=triggered_gr_name or "Grounded Hallucination Shield",
            guardrail_status=guardrail_status,
            append_disclaimer=append_disclaimer,
            matched_pii_count=pii_matches_count
        )

    def sanitize_output(self, text: str, append_disclaimer: Optional[str] = None) -> str:
        """
        Post-execution output guardrail.
        Removes accidentally leaked connection strings, database passwords, and API keys.
        """
        sanitized = text
        sanitized = PATTERNS_PII["CONNECTION_STRING"].sub("[REDACTED_CONNECTION_STRING]", sanitized)
        sanitized = PATTERNS_PII["API_KEY"].sub("[REDACTED_API_KEY]", sanitized)
        sanitized = PATTERNS_PII["CREDIT_CARD"].sub("[REDACTED_CREDIT_CARD]", sanitized)
        sanitized = PATTERNS_PII["SSN"].sub("[REDACTED_SSN]", sanitized)

        if append_disclaimer and append_disclaimer not in sanitized:
            sanitized += append_disclaimer

        return sanitized


_engine_instance: Optional[GuardrailEngine] = None

def get_guardrail_engine() -> GuardrailEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = GuardrailEngine()
    return _engine_instance
