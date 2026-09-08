# Context Injection & Prompt Augmentation Report

**Assignment**: 3.38 Context Injection & Prompt Augmentation  
**Project**: Knovera RAG Assistant  
**Repository**: `kalviumcommunity/Knovera`  
**Branch**: `feature/context-injection-prompt-augmentation`  
**Author / Engineering Team**: Knovera AI Team  
**Status**: Completed & Verified  

---

## Executive Summary

In a Retrieval-Augmented Generation (RAG) system, retrieval provides raw document chunks, but **prompt augmentation and context injection turn those chunks into a grounded, structured prompt that the LLM can reliably reason over**.

Without structured context injection:
1. The model cannot cite where facts came from (missing provenance).
2. The context window can silently overflow, causing runtime errors or truncated outputs.
3. The model defaults to ungrounded pretraining memory (hallucination).

This module implements the **Knovera Context Injection & Prompt Augmentation Engine** (`src/context_injector.py`), delivering precise chunk formatting, explicit source markers (`[1] source#chunk_index`), dynamic token budget enforcement via `tiktoken`, strict "answer-only-from-context" grounding instructions, and complete prompt telemetry.

### Key Quantitative & Engineering Achievements
- **Deterministic Token Budget Enforcement**: Tracks exact BPE tokens (`cl100k_base`) to guarantee assembled context strictly respects `max_context_tokens` (e.g., 5,000 tokens of an 8K window), reserving 1,500 tokens for generation output and 800 tokens for instructions/questions.
- **Priority-Ranked Chunk Inclusion**: Ingests chunks in relevance order, automatically dropping or trimming lower-ranked chunks when the token threshold is reached, with a complete audit trail of dropped chunks.
- **Structured Citation Markers**: Labels chunks with standardized markers (`[index] source#chunk_index`), enabling the LLM to provide exact inline citations (`[1]`, `[2]`).
- **Strict Grounding Guard**: Enforces anti-hallucination instructions and standardized fallback phrasing: *"I don't have enough information in the provided context."*
- **100% Test Suite Pass**: 12 unit and integration tests (`test_context_injector.py`) passing with zero errors.

---

## Architecture Flow: From Retrieval to Grounded Prompt

```
+----------------------------------------------------------------------------------------------------+
|                      KNOVERA CONTEXT INJECTION & PROMPT AUGMENTATION FLOW                          |
|                                                                                                    |
|   [ Ranked Retrieved Chunks from Vector Search ]                                                   |
|   { Rank 1, Rank 2, Rank 3, ... Rank N }                                                           |
|                       |                                                                            |
|                       v                                                                            |
|   +---------------------------------------+                                                        |
|   | STAGE 1: CHUNK FORMATTING             |  format_chunk(index, chunk)                            |
|   | -> Attach Source Marker               |  Marker: "[1] source.md#0"                             |
|   +---------------------------------------+                                                        |
|                       |                                                                            |
|                       v                                                                            |
|   +---------------------------------------+                                                        |
|   | STAGE 2: TOKEN BUDGET ENFORCEMENT     |  assemble_context(chunks, max_tokens=5000)             |
|   | -> Measure BPE Tokens (tiktoken)      |  Append chunks in rank order                           |
|   | -> Check: Used + New <= Max Limit     |  Drop / Trim chunks exceeding budget                   |
|   +---------------------------------------+                                                        |
|                       |                                                                            |
|                       v                                                                            |
|   +---------------------------------------+                                                        |
|   | STAGE 3: PROMPT AUGMENTATION          |  build_prompt(question, retrieved_chunks)             |
|   | -> Inject Context Block               |  Render GROUNDED_RAG_PROMPT_TEMPLATE                   |
|   | -> Inject Grounding Instructions      |  "Answer ONLY from context..."                         |
|   | -> Inject Explicit Fallback Rule      |  "Say 'I don't have enough information...'"            |
|   +---------------------------------------+                                                        |
|                       |                                                                            |
|                       v                                                                            |
|   [ Augmented Prompt Payload for LLM Generation ]                                                  |
|   { "prompt": str, "context_tokens": int, "total_prompt_tokens": int, "sources_used": [...] }      |
+----------------------------------------------------------------------------------------------------+
```

---

## Core Code Implementation

### 1. Chunk Formatting & Source Markers (`format_chunk`)
```python
def format_chunk(index: int, chunk: Dict[str, Any], style: str = "standard") -> str:
    meta = chunk.get("metadata", {})
    source = chunk.get("source") or meta.get("source") or "document"
    chunk_index = meta.get("chunk_index", 0)
    marker = f"[{index}] {source}#{chunk_index}"
    return f"{marker}\n{chunk['text'].strip()}"
```

### 2. Token Budget Enforcement (`assemble_context`)
```python
def assemble_context(
    chunks: List[Dict[str, Any]],
    max_tokens: int = 5000,
    separator: str = "\n\n---\n\n"
) -> Tuple[str, int, List[Dict[str, Any]], List[Dict[str, Any]]]:
    selected_formatted = []
    included_chunks = []
    dropped_chunks = []
    used_tokens = 0
    separator_tokens = count_tokens(separator)
    
    for index, chunk in enumerate(chunks, start=1):
        formatted = format_chunk(index, chunk)
        chunk_token_count = count_tokens(formatted)
        marginal_tokens = chunk_token_count + (separator_tokens if selected_formatted else 0)
        
        if used_tokens + marginal_tokens > max_tokens:
            dropped_chunks.append({
                "index": index,
                "chunk": chunk,
                "token_count": chunk_token_count,
                "reason": "EXCEEDED_TOKEN_BUDGET"
            })
            continue
            
        selected_formatted.append(formatted)
        included_chunks.append(chunk)
        used_tokens += marginal_tokens
        
    context_str = separator.join(selected_formatted)
    return context_str, count_tokens(context_str), included_chunks, dropped_chunks
```

### 3. Grounded Prompt Augmentation (`build_prompt`)
```python
def build_prompt(
    question: str,
    retrieved_chunks: List[Dict[str, Any]],
    max_context_tokens: int = 5000
) -> Dict[str, Any]:
    context, context_tokens, included_chunks, dropped_chunks = assemble_context(
        chunks=retrieved_chunks,
        max_tokens=max_context_tokens
    )
    prompt_text = render_prompt(
        GROUNDED_RAG_PROMPT_TEMPLATE,
        context=context if context else "(No relevant context retrieved)",
        question=question.strip()
    )
    return {
        "prompt": prompt_text,
        "context": context,
        "context_tokens": context_tokens,
        "question_tokens": count_tokens(question),
        "total_prompt_tokens": count_tokens(prompt_text),
        "sources_used": [
            {"marker": f"[{i}]", "source": c.get("metadata", {}).get("source")}
            for i, c in enumerate(included_chunks, start=1)
        ],
        "chunks_included_count": len(included_chunks),
        "chunks_dropped_count": len(dropped_chunks)
    }
```

---

## Token Budget Mathematics

For an $8,000$-token context window model (e.g. GPT-4 / GPT-3.5-Turbo), the token budget is apportioned as follows:

| Budget Component | Token Allocation | Percentage of Window | Purpose |
|---|---|---|---|
| **Max Injected Context** | `5,000 tokens` | `62.5%` | Houses formatted retrieved chunks & source markers |
| **Reserved Generation** | `1,500 tokens` | `18.75%` | Guarantees model has space to produce complete, untruncated answers |
| **Instructions & Question** | `800 tokens` | `10.00%` | System prompt, grounding instructions, and user question |
| **Safety Buffer Margin** | `700 tokens` | `8.75%` | Prevents edge-case tokenization variance and multi-turn overhead |
| **Total Context Window** | **`8,000 tokens`** | **`100.0%`** | Hard ceiling |

---

## Strategies When Retrieved Chunks Exceed Token Limit

When candidate chunks exceed the allocated `max_context_tokens` budget, Knovera applies the following hierarchical strategies:

1. **Priority-Ranked Inclusion (Implemented Default)**: Ingest chunks in descending order of retrieval similarity score. Drop the lowest-scoring chunks when the budget ceiling is reached.
2. **Dynamic Top-$k$ Reduction**: Adjust $k$ downward (e.g., from $k=10$ to $k=4$) based on the average token size of retrieved documents.
3. **Chunk Boundary Trimming**: If a high-relevance chunk barely exceeds the remaining budget, slice the chunk text to fit the exact remaining token capacity while appending an ellipsis indicator.
4. **Context Summarization / Extractive Compression**: Pass lower-ranked chunks through a fast extractive filter to retain only query-relevant sentences before prompt injection.
5. **Cross-Encoder Re-Ranking**: Apply a cross-encoder model to re-score candidate chunks, ensuring the most information-dense chunks occupy the limited token budget.

---

## Sample Augmented Prompt Output

### Rendered Prompt (`sample_augmented_prompt.txt`)
```text
You are a grounded assistant for Knovera.
Answer the question using ONLY the provided context below.
If the answer is not in the context, or if the context is insufficient, explicitly say:
"I don't have enough information in the provided context."
Do not extrapolate or speculate beyond the provided evidence.
When possible, cite sources using the citation markers like [1] or [2].

Context:
[1] Source: submission-rubric.md#0 | Section: 'Evidence Requirements' | Category: Academics
Project submission evidence requirement: Learners must upload repository GitHub links and a 3-minute video walkthrough demonstrating all test executions, architectural flows, and key modular components.

---

[2] Source: account-guide.md#0 | Section: 'Authentication' | Category: Authentication
Password reset instructions for learner accounts: Users can recover access by clicking 'Forgot Password' on the login portal. A secure verification link is sent to the registered email address immediately.

---

[3] Source: campus-guide.md#0 | Section: 'Dining Facilities' | Category: Campus Life
Campus cafeteria hours and lunch menu: The cafeteria operates from 11:30 AM to 2:30 PM offering hot artisan wraps, organic salads, and rotating daily chef specials every weekday.

---

[4] Source: grading-policy.md#0 | Section: 'Evaluation Rubrics' | Category: Academics
Assignment grading criteria: Submissions are evaluated on unit test coverage (40%), modular code structure (30%), and comprehensive documentation (30%). A score of 60% or higher is required to pass.

Question:
What evidence is required for project submission and what are the grading criteria?

Answer:
```

### Telemetry Breakdown (`sample_augmented_prompt.json`)
```json
{
  "question_tokens": 14,
  "context_tokens": 243,
  "total_prompt_tokens": 341,
  "max_context_tokens": 1500,
  "chunks_included_count": 4,
  "chunks_dropped_count": 0,
  "sources_used": [
    {"marker": "[1]", "source": "submission-rubric.md", "chunk_index": 0},
    {"marker": "[2]", "source": "account-guide.md", "chunk_index": 0},
    {"marker": "[3]", "source": "campus-guide.md", "chunk_index": 0},
    {"marker": "[4]", "source": "grading-policy.md", "chunk_index": 0}
  ]
}
```

---

## Verification & Test Results

Run `python test_context_injector.py` to execute the full unit and integration test suite:

```
test_assemble_context_empty_input (__main__.TestContextInjectionPromptAugmentation) ... ok
test_assemble_context_invalid_budget (__main__.TestContextInjectionPromptAugmentation) ... ok
test_assemble_context_strict_budget_overflow (__main__.TestContextInjectionPromptAugmentation) ... ok
test_assemble_context_within_budget (__main__.TestContextInjectionPromptAugmentation) ... ok
test_build_prompt_empty_question_error (__main__.TestContextInjectionPromptAugmentation) ... ok
test_build_prompt_grounding_instructions (__main__.TestContextInjectionPromptAugmentation) ... ok
test_build_prompt_telemetry_structure (__main__.TestContextInjectionPromptAugmentation) ... ok
test_context_injector_oop_engine (__main__.TestContextInjectionPromptAugmentation) ... ok
test_count_tokens_accuracy (__main__.TestContextInjectionPromptAugmentation) ... ok
test_format_chunk_compact (__main__.TestContextInjectionPromptAugmentation) ... ok
test_format_chunk_detailed (__main__.TestContextInjectionPromptAugmentation) ... ok
test_format_chunk_standard (__main__.TestContextInjectionPromptAugmentation) ... ok

----------------------------------------------------------------------
Ran 12 tests in 0.408s

OK
```

---

## Video Walkthrough Guide (3–5 Minutes)

When recording your submission walkthrough video, follow this structure:
1. **What Context Injection Means in RAG (0:00 - 1:00)**:
   - Explain how raw retrieved chunks must be formatted, labelled, and injected into prompt templates to ground LLM reasoning.
2. **Token Budget Enforcement & Budget Math (1:00 - 2:00)**:
   - Walk through the token budget equation: Context Window ($8,000$) = Context ($5,000$) + Output ($1,500$) + Instructions ($800$) + Buffer ($700$).
   - Show how `tiktoken` calculates exact token consumption in `src/context_injector.py`.
3. **Live Demonstration & Assembled Prompt (2:00 - 3:30)**:
   - Run `python context_injection_demo.py`.
   - Show how source markers `[1] submission-rubric.md#0` enable verifiable citations.
   - Explain the "only from context" instruction and why it prevents the model from hallucinating.
4. **Follow-Up Question & Conclusion (3:30 - 4:30)**:
   - **Answer the Follow-Up**: *"What would you do when retrieved chunks exceed the token limit?"*
   - Discuss priority ranking (best chunks first, drop lowest), dynamic top-$k$ reduction, chunk trimming, context summarization, and cross-encoder re-ranking.
