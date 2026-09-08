# Hallucination Guardrails & Refusal Handling Report

**Assignment**: 3.41 Hallucination Guardrails & Refusal Handling  
**Project**: Knovera RAG Assistant  
**Repository**: `kalviumcommunity/Knovera`  
**Branch**: `feature/hallucination-guardrails-refusal`  
**Author / Engineering Team**: Knovera AI Team  
**Status**: Completed & Verified  

---

## Executive Summary

A Retrieval-Augmented Generation (RAG) system should not attempt to answer every query. When vector retrieval yields empty, weak, or out-of-domain context, LLMs tend to generate plausible-sounding but unsupported claims—known as **hallucinations**. In high-stakes domains (healthcare, financial advice, academic credentials, legal guidance), hallucinatory answers pose severe operational and legal risks.

This module delivers the **Knovera Hallucination Guardrails & Refusal Handling Engine** (`src/hallucination_guardrail.py`), fulfilling all 5 core assignment requirements:

1. **Weak Retrieval Signal Detection (Task 1)**: Evaluates vector retrieval quality across multiple quantitative signals including empty results (`total_chunks == 0`), low top similarity score (`top_score < 0.72`), and insufficient supporting chunks (`strong_chunks < 1`).
2. **Safe Refusal Generation (Task 2)**: Intercepts weak retrieval states *before* LLM generation, returning a standardized non-hallucinatory refusal message (*"I don't have enough reliable context to answer that."*) paired with explicit status codes (`refused_weak_context`, `refused_empty_context`, `refused_low_similarity`, `refused_insufficient_support`).
3. **Configurable Relevance Thresholds & Tuning (Task 3)**: Establishes a calibrated similarity threshold (`MIN_TOP_SCORE = 0.72`, `MIN_SUPPORTING_CHUNKS = 1`) and includes an automated threshold tuning module (`tune_guardrail_thresholds`) computing precision, recall, and F1 optimization across ground-truth evaluation datasets.
4. **Confident Grounded Answer Preservation (Task 4)**: Ensures queries backed by strong supporting context pass seamlessly to grounded generation, attaching source citations (`[1]`, `[2]`) with status `"answered"`.
5. **Sample Outputs Payload & Test Suite (Task 5)**: Exports `sample_guardrail_outputs.json`, executes an interactive CLI demonstration (`hallucination_guardrail_demo.py`), and provides a comprehensive unit test suite (`test_hallucination_guardrail.py`) passing with 100% success across 16 test cases.

---

## Architecture & Pre-Generation Guardrail Flow

The guardrail acts as an explicit decision gate positioned between **Context Retrieval** and **LLM Generation**.

```
+---------------------------------------------------------------------------------------------------------+
|                               KNOVERA HALLUCINATION GUARDRAIL WORKFLOW                                  |
|                                                                                                         |
|  [ User Query ]                                                                                         |
|        |                                                                                                |
|        v                                                                                                |
|  +------------------------------+                                                                       |
|  | Vector Retrieval / Top-K    |                                                                        |
|  +------------------------------+                                                                       |
|        |                                                                                                |
|  [ Retrieved Chunks + Similarity Scores ]                                                               |
|        |                                                                                                |
|        v                                                                                                |
|  +---------------------------------------------------------------------------------------------------+  |
|  | PRE-GENERATION GUARDRAIL DECISION GATE (evaluate_retrieval_quality)                                |  |
|  |                                                                                                   |  |
|  | Signals Checked:                                                                                  |  |
|  | 1. Is chunks list empty? -> refusal_reason: "NO_CHUNKS_RETRIEVED"                                |  |
|  | 2. Is top_score < MIN_TOP_SCORE (0.72)? -> refusal_reason: "TOP_SCORE_BELOW_THRESHOLD"            |  |
|  | 3. Are strong_chunks < MIN_SUPPORTING_CHUNKS (1)? -> refusal_reason: "INSUFFICIENT_SUPPORT"      |  |
|  | 4. Is avg_score < MIN_AVG_SCORE (0.60)? -> refusal_reason: "LOW_AVERAGE_SIMILARITY"              |  |
|  +---------------------------------------------------------------------------------------------------+  |
|        |                                                                 |                              |
|        v (WEAK CONTEXT DETECTED / IS_STRONG == FALSE)                    v (STRONG CONTEXT PASSES)      |
|  +------------------------------------------------------+   +---------------------------------------+   |
|  | Immediate Safe Refusal                               |   | Grounded Generation                   |   |
|  | Answer: "I don't have enough reliable context..."    |   | generate_grounded_answer /            |   |
|  | Sources: [] | Citations: {}                          |   | answer_with_citations                 |   |
|  | Status: "refused_low_similarity" /                   |   | Status: "answered"                    |   |
|  |         "refused_empty_context"                      |   | Includes citations [1], [2]           |   |
|  +------------------------------------------------------+   +---------------------------------------+   |
|        |                                                                 |                              |
|        +-----------------------------------+-----------------------------+                              |
|                                            |                                                            |
|                                            v                                                            |
|                    [ User Response + Retrieval Telemetry Payload ]                                     |
+---------------------------------------------------------------------------------------------------------+
```

---

## Detailed Task Implementation

### Task 1: Detect Weak Retrieval (`retrieval_is_strong` & `evaluate_retrieval_quality`)

The engine checks four quantitative retrieval signals before approving LLM generation:

```python
def retrieval_is_strong(
    chunks: List[Dict[str, Any]],
    min_top_score: float = 0.72,
    min_supporting_chunks: int = 1,
    min_avg_score: float = 0.0
) -> bool:
    if not chunks:
        return False
        
    strong_chunks = [chunk for chunk in chunks if chunk.get("score", 0.0) >= min_top_score]
    if len(strong_chunks) < min_supporting_chunks:
        return False
        
    if min_avg_score > 0.0:
        scores = [chunk.get("score", 0.0) for chunk in chunks]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        if avg_score < min_avg_score:
            return False
            
    return True
```

---

### Task 2: Return Safe Refusal Response (`guarded_answer`)

When any retrieval signal fails, `guarded_answer` returns a standardized refusal payload without invoking LLM completion:

```json
{
  "question": "What is the refund policy for a product not in this corpus?",
  "answer": "I don't have enough reliable context to answer that.",
  "sources": [],
  "citations": {},
  "status": "refused_low_similarity",
  "guardrail_triggered": true,
  "retrieval_metrics": {
    "is_strong": false,
    "total_chunks": 2,
    "top_score": 0.45,
    "strong_chunks_count": 0,
    "avg_score": 0.415,
    "refusal_reason": "TOP_SCORE_BELOW_THRESHOLD (0.4500 < 0.7200)",
    "status_code": "refused_low_similarity"
  }
}
```

---

### Task 3: Relevance Threshold Selection & Dynamic Tuning

#### Default Thresholds
- `MIN_TOP_SCORE = 0.72`: Cosine similarity cutoff for semantic relevance.
- `MIN_SUPPORTING_CHUNKS = 1`: Minimum number of chunks required to support an answer.
- `MIN_AVG_SCORE = 0.60`: Average similarity floor across retrieved chunk set.

#### Dynamic Threshold Tuning (`tune_guardrail_thresholds`)
To prevent guesswork, `tune_guardrail_thresholds` benchmarks candidate thresholds against annotated ground-truth test queries:

```python
def tune_guardrail_thresholds(evaluation_queries, candidate_thresholds=[0.50, 0.60, 0.70, 0.72, 0.75, 0.80]):
    # Computes Precision, Recall, and F1-score across candidate thresholds
    # Returns best_threshold maximizing F1 score for answerable vs unanswerable classification
```

---

### Task 4: Preserve Confident Grounded Answers

When retrieved chunks satisfy quality thresholds, the guardrail passes context cleanly to the grounded generation engine:

```json
{
  "question": "What evidence is required for project submission?",
  "answer": "Based on the provided documentation: The GitHub PR must be public, open at submission time, and pass automated unit tests with a score >= 60%. [2] Learners must submit a public GitHub repository PR link and a 3-minute video walkthrough. [1]",
  "status": "answered",
  "guardrail_triggered": false,
  "retrieval_metrics": {
    "is_strong": true,
    "total_chunks": 2,
    "top_score": 0.895,
    "strong_chunks_count": 2,
    "avg_score": 0.8535,
    "refusal_reason": "STRONG_CONTEXT",
    "status_code": "answered"
  }
}
```

---

## Refusing vs. Answering Trade-offs & High-Stakes Domain Rationale

### The Trade-off Spectrum

| Metric | Low Refusal Threshold (Permissive) | High Refusal Threshold (Conservative / Guarded) |
| :--- | :--- | :--- |
| **Risk of Hallucination** | **High**: Invents answers for weak context | **Extremely Low**: Refuses when evidence is weak |
| **User Experience (Coverage)** | High answer rate, but risks misinformation | High accuracy on answered queries; safe refusals |
| **Failure Mode** | Confident wrong answers (Hallucination) | Safe refusal ("I don't know") |
| **Target Use Case** | Creative writing, casual brainstorming | Technical documentation, education, legal, medical |

### Why Refusing is Safer in High-Stakes Domains

In high-stakes environments—such as student project evaluation criteria, medical treatment guidelines, financial regulations, or legal contracts—a confident incorrect answer causes **immediate real-world harm**:
- **Academic/Assessment Context**: A student receiving incorrect submission instructions based on a hallucinatory answer will fail their submission.
- **Medical/Health Context**: A user taking medication advice generated from weakly matched context faces health risks.
- **Financial/Legal Context**: Incorrect regulatory advice creates liability and financial penalty.

Returning `"I don't have enough reliable context to answer that."` communicates boundaries accurately and prompts the user to rephrase or consult authoritative primary sources.

---

## Verification & Benchmark Results

### 1. Interactive Demo Execution (`hallucination_guardrail_demo.py`)
- Successfully evaluated strong, low-similarity, and empty retrieval scenarios.
- Correctly generated safe refusals for low-similarity (`top_score = 0.4500`) and empty retrieval scenarios.
- Produced grounded answers with inline citations for strong retrieval (`top_score = 0.8950`).
- Exported payload to `sample_guardrail_outputs.json`.

### 2. Unit Test Suite (`test_hallucination_guardrail.py`)
```powershell
py -m unittest test_hallucination_guardrail.py
Ran 16 tests in 2.319s - OK (100% Pass Rate)
```

---

## Conclusion & Recommendations

The implementation of **Hallucination Guardrails & Refusal Handling** equips the Knovera RAG Assistant with a robust defense against ungrounded generation.

### Key Recommendations for Production:
1. **Domain-Specific Threshold Calibration**: Use `tune_guardrail_thresholds` on domain query logs to periodically recalibrate `MIN_TOP_SCORE` as new document collections are ingested.
2. **Telemetry Monitoring**: Monitor `refusal_rate` metrics in telemetry to identify knowledge base coverage gaps (queries with high frequency but recurring refusals).
3. **User Query Rephrasing Prompts**: When a refusal is returned, suggest related supported topics or prompt the user to rephrase their query.
