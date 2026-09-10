# Retrieval Evaluation & Recall Testing Report

**Assignment**: 3.36 Retrieval Evaluation & Recall Testing  
**Project**: Knovera RAG Assistant  
**Repository**: `kalviumcommunity/Knovera`  
**Author / Engineering Team**: Knovera AI Team  
**Status**: Completed & Verified  

---

## Executive Summary

Retrieval quality in Retrieval-Augmented Generation (RAG) systems must be systematically **measured, not guessed**. When a generator model produces an incomplete or inaccurate response, the primary failure mode is often retrieval failure—the retriever failed to fetch the required ground-truth document chunk within the top-$k$ context window.

This module implements an enterprise **Retrieval Evaluation & Recall Testing Engine** (`src/retrieval_evaluator.py`) designed to measure **Recall@k**, **Precision@k**, **F1-Score**, **Mean Reciprocal Rank (MRR)**, and **Hit Rate** using a ground-truth **Labelled Query Benchmark Set** (`sample_labelled_queries.json`).

### Key Quantitative Results
- **Recall@5**: **`100.0%`** (100% of all ground-truth target chunks retrieved in top-5).
- **Precision@5**: **`26.7%`** (Reflects context noise dilution when $k > |\text{relevant chunks}|$).
- **Hit Rate@5**: **`100.0%`** (6 out of 6 test queries retrieved at least 1 relevant chunk).
- **MRR (Mean Reciprocal Rank)**: **`0.9167`** (Ground-truth chunks average rank #1 across queries).
- **Multi-K Performance Trade-off**:
  - $k=1$: Recall@1 = **`75.0%`**, Precision@1 = **`83.3%`**, Passed = `4/6`
  - $k=3$: Recall@3 = **`91.7%`**, Precision@3 = **`38.9%`**, Passed = `5/6`
  - $k=5$: Recall@5 = **`100.0%`**, Precision@5 = **`26.7%`**, Passed = `6/6`
  - $k=10$: Recall@10 = **`100.0%`**, Precision@10 = **`13.3%`**, Passed = `6/6`

---

## Core Concepts & Mathematical Formulations

Retrieval quality evaluation measures whether the vector store search engine delivers the exact ground-truth information required for downstream LLM response generation.

```
+-----------------------------------------------------------------------------------+
|                            RETRIEVAL EVALUATION PIPELINE                          |
|                                                                                   |
|  [ Labelled Test Query ] ---> [ TopKRetriever / VectorDB ]                        |
|                                         |                                         |
|                                         v                                         |
|                             Retrieved Chunk IDs (Top-K)                           |
|                                         |                                         |
|                                         v                                         |
|                        +---------------------------------+                        |
|                        |       RetrievalEvaluator        |                        |
|                        |                                 |                        |
|                        |  Hits = Retrieved INTERSECT Rel |                        |
|                        |  Recall@k = Hits / |Rel|        |                        |
|                        |  Precision@k = Hits / k         |                        |
|                        |  MRR = 1 / First Hit Rank       |                        |
|                        +---------------------------------+                        |
|                                         |                                         |
|                                         v                                         |
|                   [ Failure Diagnostics & Audit Log Export ]                      |
+-----------------------------------------------------------------------------------+
```

### Mathematical Definitions

1. **Recall@k**: Measures the fraction of ground-truth relevant chunks present in the top-$k$ retrieved results.
   $$\text{Recall}@k = \frac{|\text{Retrieved Top-}k \cap \text{Relevant}|}{|\text{Relevant}|}$$

2. **Precision@k**: Measures the proportion of retrieved top-$k$ chunks that are actually relevant.
   $$\text{Precision}@k = \frac{|\text{Retrieved Top-}k \cap \text{Relevant}|}{|\text{Retrieved Top-}k|}$$

3. **F1-Score**: Harmonic mean of Precision and Recall, balancing coverage against context overcrowding.
   $$\text{F1-Score} = \frac{2 \cdot \text{Precision}@k \cdot \text{Recall}@k}{\text{Precision}@k + \text{Recall}@k}$$

4. **Mean Reciprocal Rank (MRR)**: Evaluates ranking quality by measuring the reciprocal rank of the first relevant chunk.
   $$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$

5. **Hit Rate@k**: Binary indicator showing whether at least one relevant chunk was retrieved within top-$k$.
   $$\text{Hit Rate}@k = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \mathbb{I}(|\text{Hits}_i| > 0)$$

---

## Labelled Query Benchmark Dataset Design

A trustworthy labelled query set maps realistic user questions to exact ground-truth chunk IDs (`relevant_chunk_ids`). In `sample_labelled_queries.json`, test queries span 6 key domain categories:

```json
[
  {
    "query": "How can a learner reset their password?",
    "relevant_chunk_ids": ["account-guide.md:0", "account-guide.md:1"],
    "category": "Authentication"
  },
  {
    "query": "What evidence is required for project submission?",
    "relevant_chunk_ids": ["submission-rubric.md:0"],
    "category": "Academics"
  },
  {
    "query": "When does the cafeteria menu change?",
    "relevant_chunk_ids": ["campus-guide.md:0"],
    "category": "Campus Life"
  },
  {
    "query": "What are the API rate limits and token quotas for vector indexing?",
    "relevant_chunk_ids": ["api-policy.md:0", "api-policy.md:1"],
    "category": "Operations"
  },
  {
    "query": "What is the refund eligibility window for enterprise billing tiers?",
    "relevant_chunk_ids": ["billing-faq.md:0"],
    "category": "Billing"
  },
  {
    "query": "How do vector databases handle HNSW distance metrics?",
    "relevant_chunk_ids": ["vector-db-guide.md:0"],
    "category": "Infrastructure"
  }
]
```

---

## Code Architecture & Implementation

The module `src/retrieval_evaluator.py` provides clean dataclasses and an evaluator engine:

- **`LabelledQuery`**: Dataclass representing a test query, category, description, and ground-truth `relevant_chunk_ids` set.
- **`QueryEvaluationResult`**: Dataclass holding per-query retrieved IDs, hits, recall, precision, F1, MRR, scores, and diagnosed failure cause.
- **`AggregateMetrics`**: Dataclass capturing dataset-level summary statistics across all queries.
- **`RetrievalEvaluator`**: Core evaluator class implementing:
  - `evaluate_query(query_item, k=5, collection_name=None)`
  - `evaluate_queries(labelled_queries, k=5, collection_name=None)`
  - `evaluate_across_k(labelled_queries, k_values=[1, 3, 5, 10])`
  - `inspect_failures(aggregate_metrics)`
  - `export_results(aggregate_metrics, json_path, txt_path)`

---

## Experimental Evaluation Results

### Multi-K Parameter Analysis ($k=1, 3, 5, 10$)

| Top-K Cutoff ($k$) | Avg Recall@k | Avg Precision@k | Avg F1-Score | MRR | Hit Rate@k | Passed Queries (Recall=1.0) |
|---|---|---|---|---|---|---|
| **$k=1$** | **`75.0%`** | **`83.3%`** | **`0.7778`** | `0.8333` | `83.3%` | `4 / 6` |
| **$k=3$** | **`91.7%`** | **`38.9%`** | **`0.5333`** | `0.9167` | `100.0%` | `5 / 6` |
| **$k=5$** | **`100.0%`** | **`26.7%`** | **`0.4127`** | `0.9167` | `100.0%` | `6 / 6` |
| **$k=10$** | **`100.0%`** | **`13.3%`** | **`0.2323`** | `0.9167` | `100.0%` | `6 / 6` |

### Key Observations & Trade-Offs:
1. **$k=1$ Cutoff**: High Precision (83.3%) but insufficient Recall (75.0%). Queries with multi-chunk ground-truth requirements (e.g. password recovery with 2 chunks) fail to achieve full recall.
2. **$k=3$ Cutoff**: Reaches 91.7% Recall and 100% Hit Rate while maintaining higher precision than $k=5$.
3. **$k=5$ Cutoff (Optimal for Recall)**: Achieves **100% Recall** across all test queries, ensuring zero missing ground-truth chunks for generator context.
4. **$k=10$ Cutoff**: Does not increase Recall beyond 100%, but severely degrades Precision (13.3%), introducing distractor noise into LLM prompts.

---

## Failure Inspection & Root Cause Diagnosis

When evaluating at restrictive cutoffs (e.g., $k=1$), `inspect_failures()` categorizes low-recall cases:

```
--------------------------------------------------------------------------------
 FAILURE DIAGNOSTIC REPORT (k=1 Cutoff)
--------------------------------------------------------------------------------
Failure #1:
  Query:            'How can a learner reset their password?'
  Recall@1:         50.0% (Retrieved 1 of 2 expected chunks)
  Expected Chunks:  ['account-guide.md:0', 'account-guide.md:1']
  Retrieved Chunks: ['account-guide.md:0']
  Diagnosed Cause:  TOO_SMALL_K: Relevant target set size (2) exceeds top-k cutoff (1).
  Recommendation:   Increase top-k parameter from 1 to at least 2.

Failure #2:
  Query:            'What are the API rate limits and token quotas for vector indexing?'
  Recall@1:         0.0% (Distractor chunk returned at Rank #1)
  Expected Chunks:  ['api-policy.md:0', 'api-policy.md:1']
  Retrieved Chunks: ['distractor-guide.md:0']
  Diagnosed Cause:  ZERO_HITS_EMBEDDING_MISMATCH: Target relevant chunk was not present in top-k results.
  Recommendation:   Implement hybrid search (vector + BM25 keyword matching) or query rewriting/expansion.
```

### Common Failure Modes & Diagnoses:
1. **`TOO_SMALL_K`**: Ground-truth target set size ($|\text{relevant}|$) exceeds top-$k$ cutoff.
2. **`ZERO_HITS_EMBEDDING_MISMATCH`**: Dense embedding similarity score is lower than distractor chunks due to query wording mismatch.
3. **`RANKING_NOISE_OVERCROWDING`**: Relevant chunk exists in vector store, but is pushed outside top-$k$ by irrelevant chunks.

---

## Actionable Roadmap for Improving Recall

To systematically improve Recall@k in production RAG pipelines, implement the following prioritized optimizations:

1. **Tune Top-K Cutoff**: Set $k=5$ as the baseline for multi-chunk queries.
2. **Hybrid Search (BM25 + Dense Vectors)**: Combine semantic embeddings with lexical keyword matching to fix vocabulary mismatch failures.
3. **Two-Stage Cross-Encoder Re-Ranking**: Retrieve top-20 candidate chunks with vector search, then re-rank using a Cross-Encoder model before cutting off at top-5.
4. **Metadata Filtering**: Pre-filter search space by domain category (`Authentication`, `Academics`, `Operations`) to eliminate distractor noise.
5. **Adjust Chunking Granularity & Overlap**: Ensure chunk boundaries do not split critical context sentences across chunk boundaries.

---

## Verification & Test Suite Execution

The module is verified via pytest test suite `test_retrieval_evaluator.py` and demonstration script `retrieval_evaluation_demo.py`.

```bash
# Run retrieval evaluation test suite
pytest test_retrieval_evaluator.py -v

# Run full project pytest suite
pytest -v

# Run end-to-end evaluation demonstration
py retrieval_evaluation_demo.py
```

### Test Summary:
- `test_retrieval_evaluator.py`: **6 / 6 Passed**
- Full Workspace Test Suite: **83 / 83 Passed**
- Generated Outputs:
  - `outputs/retrieval_evaluation_results.json`
  - `outputs/retrieval_evaluation_audit.log`
