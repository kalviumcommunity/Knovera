"""
src/services/semantic_engine.py

Semantic Intent Normalizer, Synonym Expander, and Advanced Grounded Synthesizer for Knovera.
Solves semantic equivalence between inquiry directives such as "describe", "define",
"explain", "what is", and provides deep extractive semantic synthesis from retrieved context.
"""

import re
import logging
from typing import List, Dict, Any, Tuple, Set, Optional

logger = logging.getLogger("knovera.semantic")

# Equivalence clusters for common inquiry directives
INQUIRY_DIRECTIVES: Set[str] = {
    "describe", "define", "definition", "explain", "explanation", "outline",
    "clarify", "delineate", "summarize", "summary", "detail", "details",
    "what", "is", "are", "tell", "me", "about", "give", "overview"
}

# Domain synonym mappings for query expansion and semantic matching
DOMAIN_SYNONYMS: Dict[str, Set[str]] = {
    "describe": {"define", "definition", "explain", "explanation", "outline", "detail", "overview", "policy"},
    "define": {"describe", "definition", "explain", "meaning", "outline", "terms", "criteria", "policy"},
    "refund": {"refunds", "reimbursement", "return", "credit", "chargeback", "money-back"},
    "policy": {"policies", "guideline", "guidelines", "rules", "standards", "protocol", "matrix"},
    "retention": {"retain", "retention", "storage", "stored", "persist", "purge", "deletion", "timeline"},
    "downtime": {"outage", "unavailable", "incident", "disruption", "service-credit", "sla"},
    "pricing": {"price", "prices", "cost", "costs", "tier", "tiers", "subscription", "fee", "fees"},
    "sla": {"uptime", "availability", "service level", "credit", "downtime"},
    "onboarding": {"setup", "training", "provisioning", "activation"},
    "compliance": {"gdpr", "ccpa", "audit", "security", "privacy", "regulation"}
}

# Noise phrases to strip when isolating the core topical entities
NOISE_PREFIXES = [
    r'^can\s+you\s+(please\s+)?(describe|define|explain|tell\s+me\s+about)\s+(the\s+)?',
    r'^(please\s+)?(describe|define|explain|tell\s+me\s+about|what\s+is|what\s+are)\s+(the\s+)?',
    r'^(i\s+would\s+like\s+to\s+know|could\s+you\s+clarify)\s+(the\s+)?'
]


def extract_core_topic(query: str) -> str:
    """
    Strips inquiry scaffolding to isolate core topical entities (e.g. 'refund policy').
    """
    clean_q = query.strip()
    for pat in NOISE_PREFIXES:
        clean_q = re.sub(pat, '', clean_q, flags=re.IGNORECASE).strip()
    return clean_q.rstrip("?.! ") or query


FOLLOW_UP_PATTERNS = [
    r'\b(this|that|it|these|those)\b',
    r'\b(more\s+about|elaborate|more\s+details|tell\s+me\s+more|expand\s+on|continue|what\s+else|explain\s+further)\b',
    r'^(can\s+you\s+)?(tell\s+me\s+)?more(\s+about\s+this)?\??$',
    r'^(can\s+you\s+)?elaborate(\s+on\s+this)?\??$',
    r'^(what\s+about\s+it|how\s+come|why\s+is\s+that|give\s+more\s+details)\??$',
]


def is_conversational_follow_up(query: str) -> bool:
    """
    Detects if a user query is a follow-up continuation referencing prior context
    using pronouns (this, that, it) or elaboration phrasing.
    """
    q_clean = query.strip().lower()
    for pat in FOLLOW_UP_PATTERNS:
        if re.search(pat, q_clean):
            return True
    return False


def reformulate_conversational_query(
    current_query: str,
    history: Optional[List[Any]] = None
) -> str:
    """
    Resolves conversational pronouns and follow-up continuations by contextualizing
    the query with the core topic from recent conversation history turns.
    """
    if not history or not is_conversational_follow_up(current_query):
        return current_query

    # Search history backwards for the most recent user question or topical entity
    last_topic = None
    for turn in reversed(history):
        role = getattr(turn, 'role', None) or (turn.get('role') if isinstance(turn, dict) else None)
        content = getattr(turn, 'content', None) or (turn.get('content') if isinstance(turn, dict) else None)

        if role == 'user' and content and content.strip():
            candidate = extract_core_topic(content.strip())
            # Ensure candidate is not itself just a follow-up pronoun
            if candidate and not is_conversational_follow_up(candidate):
                last_topic = candidate
                break

    if not last_topic:
        # Check assistant turns for policy / document names if no clean user topic
        for turn in reversed(history):
            role = getattr(turn, 'role', None) or (turn.get('role') if isinstance(turn, dict) else None)
            content = getattr(turn, 'content', None) or (turn.get('content') if isinstance(turn, dict) else None)
            if role == 'assistant' and content:
                bold_matches = re.findall(r'\*\*([^*]+)\*\*', content)
                if bold_matches:
                    last_topic = bold_matches[0]
                    break

    if last_topic:
        # Contextualize: e.g. "refund policy: can you tell more about this"
        return f"{last_topic}: {current_query}"

    return current_query


def expand_query_synonyms(query: str) -> Set[str]:
    """
    Expands words in the query with their domain synonyms.
    """
    tokens = [w.lower() for w in re.findall(r'\b\w+\b', query) if len(w) > 1]
    expanded = set(tokens)
    for t in tokens:
        for root_key, syn_set in DOMAIN_SYNONYMS.items():
            if t == root_key or t in syn_set:
                expanded.add(root_key)
                expanded.update(syn_set)
    return expanded


def synthesize_semantic_grounded_answer(
    query: str,
    context: str,
    chunks: Optional[List[Dict[str, Any]]] = None
) -> str:
    """
    Synthesizes a structured, grounded answer from retrieved context passages.
    Understands semantic intent (e.g., 'describe' and 'define') and extracts
    meaningful substantive sentences while filtering out trivial document headers.
    """
    if not context or not context.strip():
        return "I don't have enough reliable context to answer that."

    core_topic = extract_core_topic(query)
    expanded_query_words = expand_query_synonyms(query)
    topic_words = set(w.lower() for w in re.findall(r'\b\w+\b', core_topic) if len(w) > 2)

    # Check if query is an inquiry/definition request
    query_lower = query.lower()
    is_definition_request = any(d in query_lower for d in ["describe", "define", "what is", "explain", "overview"])

    # Parse context into source blocks
    blocks = context.split("\n\n")
    candidate_sentences: List[Tuple[float, str, str]] = []

    for block in blocks:
        lines = [line.strip() for line in block.split("\n") if line.strip()]
        if not lines:
            continue

        header = lines[0]
        citation = header.split("]")[0] + "]" if "]" in header else "[1]"
        body = " ".join(lines[1:]) if len(lines) > 1 else lines[0]

        # Break body into natural sentences
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', body) if len(s.strip()) > 15]

        for sent in sentences:
            sent_lower = sent.lower()

            # Penalize trivial metadata lines (headers, IDs, dates)
            if any(meta_word in sent_lower for meta_word in ["document id:", "classification:", "applies to:", "audience:", "version:"]):
                continue

            sent_words = set(w.lower() for w in re.findall(r'\b\w+\b', sent_lower))
            
            # 1. Topic word overlap (highest weight)
            topic_overlap = len(topic_words.intersection(sent_words))
            
            # 2. Synonym expansion overlap
            syn_overlap = len(expanded_query_words.intersection(sent_words))
            
            # 3. Definition/explanation bonus
            def_bonus = 0.0
            if is_definition_request:
                if any(k in sent_lower for k in ["is defined as", "outlines", "eligible for", "policy", "window", "timeline", "guarantees", "covers", "includes"]):
                    def_bonus = 1.5
                if any(char.isdigit() for char in sent):
                    # Sentences with specific numbers/timelines/percentages are highly informative
                    def_bonus += 0.8

            score = (topic_overlap * 3.0) + (syn_overlap * 1.0) + def_bonus

            if score > 1.0:
                clean_sent = sent.rstrip(".") + f". {citation}"
                candidate_sentences.append((score, clean_sent, citation))

    if candidate_sentences:
        # Sort sentences by score descending
        candidate_sentences.sort(key=lambda x: x[0], reverse=True)
        
        # Deduplicate while preserving order
        seen = set()
        chosen = []
        for s in candidate_sentences:
            normalized = re.sub(r'\[\d+\]', '', s[1]).strip()[:80]
            if normalized not in seen:
                seen.add(normalized)
                chosen.append(s[1])
            if len(chosen) >= 4:
                break

        # Format output cleanly
        clean_topic = core_topic.split(":")[0].strip() if ":" in core_topic else core_topic
        if len(chosen) == 1:
            return f"Based on the official documentation: {chosen[0]}"
        else:
            intro = f"Based on Knovera's verified documentation regarding **{clean_topic}**:"
            bullets = "\n".join([f"- {sentence}" for sentence in chosen])
            return f"{intro}\n\n{bullets}"

    # Fallback to first informative line from context
    first_block = blocks[0] if blocks else ""
    return f"According to the documentation: {first_block[:250]}..."
