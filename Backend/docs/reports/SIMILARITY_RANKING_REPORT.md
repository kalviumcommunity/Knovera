# Assignment 3.27: Embedding Similarity & Distance Metrics

## Executive Summary

This report documents the design, implementation, mathematical analysis, and empirical verification for **Assignment 3.27: Embedding Similarity & Distance Metrics** within the **Knovera RAG Assistant** platform.

Generating vector embeddings is only the first step in Retrieval-Augmented Generation (RAG). To retrieve relevant context for a user prompt, the system must project the query into vector space and compute geometric proximity against stored chunk embeddings. This report details our implementation of vector similarity and distance metrics ([`src/embedding_generator.py`](file:///d:/Project/Knovera/src/embedding_generator.py)), an interactive ranking demonstration ([`similarity_ranking_demo.py`](file:///d:/Project/Knovera/similarity_ranking_demo.py)), automated unit test coverage ([`test_similarity_ranking.py`](file:///d:/Project/Knovera/test_similarity_ranking.py)), and empirical audit outputs ([`outputs/similarity_ranking_output.txt`](file:///d:/Project/Knovera/outputs/similarity_ranking_output.txt) and [`sample_similarity_rankings.json`](file:///d:/Project/Knovera/sample_similarity_rankings.json)).

---

## Architectural Breakdown & Implementation

```
+---------------------------------------------------------------------------------------------------+
|                            Knovera Similarity & Ranking Engine                                    |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|  User Query: "How can a learner reset their password?"                                           |
|       |                                                                                           |
|       v                                                                                           |
|  [Embedding Engine] ----> Query Vector q (1536 dimensions)                                        |
|                                 |                                                                 |
|                                 v                                                                 |
|  [Stored Corpus Records] ----> [Metric Engine]                                                    |
|  - Chunk 0: Account Guide         - Cosine Similarity:  cos(theta) = (q . d) / (||q|| * ||d||)    |
|  - Chunk 1: Email Recovery        - Cosine Distance:    1 - cos(theta)                          |
|  - Chunk 2: Cafeteria Menu        - Dot Product:        q . d                                     |
|  - Chunk 3: Technical SLA         - Euclidean Dist L2:  ||q - d||_2                               |
|  - Chunk 4: Token Chunking               |                                                        |
|                                          v                                                        |
|                                [Ranking & Sorting]                                                |
|                                (Descending for Similarity, Ascending for Distance)                |
|                                          |                                                        |
|                                          v                                                        |
|  [Retrieved Payload] <-------------------+                                                        |
|  {                                                                                                |
|     "query": "How can a learner reset their password?",                                           |
|     "metric": "cosine",                                                                           |
|     "most_similar":  { "rank": 1, "score": 0.9967, "source": "account-guide.md" },                 |
|     "least_similar": { "rank": 5, "score": -0.0376, "source": "rag-architecture.md" }             |
|  }                                                                                                |
+---------------------------------------------------------------------------------------------------+
```

### 1. Multi-Metric Engine (`src/embedding_generator.py`)
We extended `src/embedding_generator.py` with standalone functions and engine dispatch logic for 5 core geometric metrics:

- **Cosine Similarity**: $\cos(\theta) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\|_2 \|\mathbf{b}\|_2}$. Range: $[-1.0, 1.0]$. Higher is more similar.
- **Cosine Distance**: $D_{\text{cos}}(\mathbf{a}, \mathbf{b}) = 1.0 - \text{Cosine Similarity}(\mathbf{a}, \mathbf{b})$. Range: $[0.0, 2.0]$. Lower is closer.
- **Dot Product**: $\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{D} a_i b_i$. Range: $(-\infty, \infty)$. Equal to Cosine Similarity when vectors are $L_2$-normalized.
- **Euclidean Distance ($L_2$)**: $\|\mathbf{a} - \mathbf{b}\|_2 = \sqrt{\sum_{i=1}^{D} (a_i - b_i)^2}$. Range: $[0.0, \infty)$. Lower is closer.
- **Manhattan Distance ($L_1$)**: $\|\mathbf{a} - \mathbf{b}\|_1 = \sum_{i=1}^{D} |a_i - b_i|$. Range: $[0.0, \infty)$. Lower is closer.

### 2. Chunk Ranking Pipeline (`EmbeddingGenerator.rank_chunks`)
The `rank_chunks` method accepts a query string or vector, embeds the query with the active embedding model, calculates scores against all stored records, assigns 1-indexed ranks, sorts records based on metric type (descending for similarity, ascending for distance), and extracts `most_similar` and `least_similar` chunks with complete metadata (`source`, `chunk_index`, `category`, `section`).

---

## Task Solutions & Empirical Results

### Task 1 - Compute Similarity Metrics
Using `cosine_similarity`, `cosine_distance`, `dot_product`, `euclidean_distance`, and `manhattan_distance`, we evaluated pairwise metrics across sample text vectors:

| Comparison Pair | Cosine Sim ($\uparrow$) | Cosine Dist ($\downarrow$) | Dot Product ($\uparrow$) | Euclidean $L_2$ ($\downarrow$) | Manhattan $L_1$ ($\downarrow$) |
|---|---|---|---|---|---|
| **Similar Pair** (*Password reset* vs *Recover access*) | **`0.9939`** | **`0.0061`** | **`0.9939`** | **`0.1104`** | **`3.4436`** |
| **Dissimilar Pair** (*Password reset* vs *Cafeteria menu*) | **`-0.0088`** | **`1.0088`** | **`-0.0088`** | **`1.4204`** | **`43.9608`** |

> [!NOTE]
> Notice that for $L_2$-normalized embedding vectors, Cosine Similarity and Dot Product produce identical numerical values (`0.9939`), while Cosine Distance equals $1.0 - 0.9939 = 0.0061$.

### Task 2 & 3 - Compare Query Against Chunks, Rank & Show Results
Sample Query: `"How can a learner reset their password?"`

We compared the query embedding against 5 corpus chunks across 4 distinct domains:

```text
============================================================
 MOST SIMILAR CHUNK (Rank #1, Score: 0.9967)
============================================================
  Source File : account-guide.md
  Chunk Index : 0
  Category    : Authentication
  Section     : Password Recovery
  Text        : "Password reset instructions for learner accounts: Users can recover access by clicking 'Forgot Password' on the portal."

============================================================
 LEAST SIMILAR CHUNK (Rank #5, Score: -0.0376)
============================================================
  Source File : rag-architecture.md
  Chunk Index : 5
  Category    : Engineering
  Section     : Token Chunking
  Text        : "Python token chunking uses tiktoken cl100k_base encoding with sliding window overlap to preserve semantic context."
```

#### Complete Retrieval Ranking Table (Cosine Similarity)

| Rank | Score | Source File | Category | Section | Snippet Preview |
|---|---|---|---|---|---|
| **#1** | **`0.9967`** | `account-guide.md` | Authentication | Password Recovery | *"Password reset instructions for learner accounts..."* |
| **#2** | **`0.9916`** | `account-guide.md` | Authentication | Email Verification | *"Learners can recover access using registered email..."* |
| **#3** | **`0.9695`** | `sla-policy.md` | Support SLA | Service Levels | *"Tier 1 Technical Support responds within 2 hours..."* |
| **#4** | **`-0.0062`** | `campus-guide.md` | Campus Life | Dining Options | *"The cafeteria menu changes every Friday..."* |
| **#5** | **`-0.0376`** | `rag-architecture.md` | Engineering | Token Chunking | *"Python token chunking uses tiktoken cl100k_base..."* |

---

## Task 4 - Technical Justification (Why Cosine Similarity?)

Cosine Similarity is the preferred metric for RAG text retrieval due to 4 primary properties:

1. **Directional Invariance**: Cosine similarity measures the angle $\theta$ between vectors in high-dimensional space ($\mathbb{R}^{1536}$). It evaluates semantic orientation rather than magnitude.
2. **Length & Norm Independence**: In NLP, longer document chunks or repeated vocabulary can inflate raw vector norms ($\|\mathbf{d}\|_2$). Euclidean distance can penalize long documents even if they match the query's topic perfectly. Cosine similarity normalizes vector lengths to $1.0$, preventing document length bias.
3. **Fixed Boundary Limits**: Cosine similarity is strictly bounded in $[-1.0, 1.0]$. This makes thresholding intuitive (e.g., setting a cutoff score of $\ge 0.70$ for context inclusion).
4. **Equivalency to Dot Product for L2-Normalized Vectors**: Standard embedding models (such as OpenAI's `text-embedding-3-small` or HuggingFace `bge-small`) return $L_2$-normalized unit vectors ($\|\mathbf{v}\|_2 = 1.0$). For unit vectors:
   $$\text{Cosine Similarity}(\mathbf{q}, \mathbf{d}) = \frac{\mathbf{q} \cdot \mathbf{d}}{1.0 \times 1.0} = \mathbf{q} \cdot \mathbf{d}$$
   This allows vector databases (Pinecone, Qdrant, Milvus, FAISS) to substitute cosine similarity with SIMD-accelerated dot product matrix multiplication.

---

## Cross-Metric Ranking Consistency Proof

To demonstrate that retrieval order is preserved across metric representations, we ran all 4 metrics against our corpus for query `"How can a learner reset their password?"`:

| Metric | Metric Type | Score Range | Rank #1 Source | Rank #1 Score | Rank #5 Source | Rank #5 Score |
|---|---|---|---|---|---|---|
| **Cosine Similarity** | Similarity ($\uparrow$) | `[-1.0, 1.0]` | `account-guide.md` | `0.9967` | `rag-architecture.md` | `-0.0376` |
| **Cosine Distance** | Distance ($\downarrow$) | `[0.0, 2.0]` | `account-guide.md` | `0.0033` | `rag-architecture.md` | `1.0376` |
| **Dot Product** | Similarity ($\uparrow$) | `[-1.0, 1.0]` | `account-guide.md` | `0.9967` | `rag-architecture.md` | `-0.0376` |
| **Euclidean ($L_2$)** | Distance ($\downarrow$) | `[0.0, inf)` | `account-guide.md` | `0.0815` | `rag-architecture.md` | `1.4406` |

> [!IMPORTANT]
> All 4 metrics produce the exact same top-to-bottom rank ordering (`account-guide.md#0` $\rightarrow$ `account-guide.md#1` $\rightarrow$ `sla-policy.md#2` $\rightarrow$ `campus-guide.md#3` $\rightarrow$ `rag-architecture.md#5`).

---

## Deep Dive: What a High Similarity Score Guarantees vs. Does Not Guarantee

Understanding the limitations of vector similarity is vital for designing robust production RAG architectures.

### What a High Similarity Score GUARANTEES
- **Geometric Proximity**: The query vector and chunk vector point in nearly the same direction within the embedding model's latent vector space.
- **Topical Semantic Overlap**: The chunk discusses concepts, entities, or topics closely related to the terminology and context of the query.
- **High Candidate Relevance**: The chunk is statistically the best candidate context within the index to provide background for the user query.

### What a High Similarity Score DOES NOT GUARANTEE
- **Factual Accuracy**: A chunk stating *"Learners reset passwords by mailing physical letters to support"* may have high embedding similarity to a password reset query, even though the procedure is entirely false or outdated.
- **Temporal Freshness**: Embeddings do not encode timestamp currency. A 2018 policy chunk and a 2026 policy chunk may score identically.
- **Completeness**: A high-scoring chunk might only contain a partial sentence or fragment, lacking the full answer needed by the LLM.
- **Hallucination / Safety Immunity**: High similarity does not guarantee the content is safe, unbiased, or free of prompt injection attacks.
- **Answer Guarantee**: Vector retrieval selects candidate context; downstream prompt engineering, reranking, metadata filtering, and LLM verification are still required to generate accurate answers.

---

## Video Script Outline (3-5 Minutes)

### Slide / Scene Breakdown

1. **Introduction & Overview (0:00 - 0:45)**
   - Introduce yourself and state the goal: demonstrating embedding similarity and chunk ranking for query retrieval in Knovera.
   - Explain why vector search is needed: keyword search fails when word phrasing differs, whereas embeddings map semantic meaning into geometric space.

2. **What Cosine Similarity Measures & Similarity vs Distance (0:45 - 1:45)**
   - Explain Cosine Similarity: $\cos(\theta) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|}$. It measures the angle/direction between vectors.
   - Contrast Similarity vs Distance: Similarity asks *"How close are these meanings?"* (higher is better, max 1.0). Distance asks *"How far apart are they?"* (lower is better, min 0.0).
   - Explain why Cosine Distance is simply $1 - \text{Similarity}$.

3. **Code Walkthrough & Live Ranking Demo (1:45 - 3:15)**
   - Show `similarity_ranking_demo.py` and `src/embedding_generator.py`.
   - Run the demo script live in terminal.
   - Walk through query `"How can a learner reset their password?"`.
   - Highlight Rank #1 (Most Similar): `account-guide.md` (Score: `0.9967`).
   - Highlight Rank #5 (Least Similar): `rag-architecture.md` (Score: `-0.0376`).
   - Show cross-metric table proving Cosine, Cosine Distance, Dot Product, and Euclidean distance all agree on ranking.

4. **Follow-Up Question: High Score Guarantees vs Non-Guarantees (3:15 - 4:30)**
   - Explicitly answer: *"What does a high similarity score guarantee and what does it NOT guarantee?"*
   - **Guarantees**: Topic similarity and vector proximity in embedding space.
   - **Does NOT guarantee**: Factuality, truthfulness, up-to-date freshness, completeness, or safety against hallucinations.

5. **Conclusion & GitHub PR (4:30 - 5:00)**
   - Summarize findings and show unit test execution (`py -m unittest test_similarity_ranking.py`).
   - Mention public GitHub PR and conclude.

---

## Verification & Output Artifacts

- **Core Module**: [`src/embedding_generator.py`](file:///d:/Project/Knovera/src/embedding_generator.py)
- **Interactive Demo**: [`similarity_ranking_demo.py`](file:///d:/Project/Knovera/similarity_ranking_demo.py)
- **Unit Test Suite**: [`test_similarity_ranking.py`](file:///d:/Project/Knovera/test_similarity_ranking.py)
- **Execution Log**: [`outputs/similarity_ranking_output.txt`](file:///d:/Project/Knovera/outputs/similarity_ranking_output.txt)
- **JSON Export**: [`sample_similarity_rankings.json`](file:///d:/Project/Knovera/sample_similarity_rankings.json)
- **Technical Notes**: [`prompts/similarity_ranking_notes.md`](file:///d:/Project/Knovera/prompts/similarity_ranking_notes.md)
