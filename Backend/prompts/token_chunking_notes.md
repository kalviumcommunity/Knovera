# Concept 3.23: Token-Aware Chunk Sizing & Overlap Notes

## 1. Core Concept Overview
In RAG (Retrieval-Augmented Generation), chunk sizing by character count is fundamentally flawed because models and embedding endpoints operate on **tokens**, not characters.

* **Density Problem**:
  * 500 characters of dense technical JSON / code can easily exceed 180+ tokens.
  * 500 characters of plain English prose is only ~90-110 tokens.
* **Boundary Problem**:
  * A hard cut without overlap risks slicing sentences, equations, and answers clean in half, causing downstream LLM hallucinations or failure to retrieve relevant context.

---

## 2. Technical Implementation in Knovera
* **Tokenizer**: `tiktoken` with `cl100k_base` encoding (OpenAI standard for GPT-4o, GPT-3.5, and `text-embedding-3-small`).
* **Chunker Class**: [`src/token_chunker.py`](../src/token_chunker.py)
* **Algorithm**:
  $$\text{step} = \text{chunk\_size} - \text{overlap}$$
  The sliding window advances by `step` tokens each cycle, ensuring that the last $N$ tokens of chunk $k$ are repeated at the beginning of chunk $k+1$.

---

## 3. Justification of Parameters for Target Model

### Target Architecture
* **Vector Embeddings**: `text-embedding-3-small` (1536 dims, max input: 8,191 tokens).
* **Generator Model**: `GPT-4o-mini` (128K context window) / `GPT-3.5-Turbo` (16K context window).

### Chosen Settings: 400 Tokens Chunk Size & 60 Tokens Overlap (15%)
1. **Semantic Completeness**: 400 tokens (~300 words) captures complete business rules, policies, and code snippets.
2. **Context Window Math**:
   * System Prompt: ~250 tokens
   * User Query & History: ~200 tokens
   * Top-$k$ Retrieved Chunks ($k = 4$): $4 \times 400 = 1,600$ tokens
   * LLM Output Generation: ~500 tokens
   * **Total Context Footprint**: **~2,550 tokens** (well within context budget limits).
3. **15% Overlap Balance**: 60 tokens comfortably encompasses standard 15–25 word sentences (~20–35 tokens), guaranteeing 100% boundary safety with negligible (~9–15%) storage overhead.

---

## 4. Video Walkthrough Script Outline (3–5 Minutes)

### Section 1: Introduction & Why Token-Based Sizing? (1 min)
* "Hello! In this walkthrough, I am demonstrating Concept 3.23: Token-Aware Chunk Sizing and Overlap for the Knovera RAG Assistant."
* "Character-based chunking cannot guarantee model budget compliance because token density varies wildly between dense code/JSON and plain text. Token-based sizing with `tiktoken` guarantees exact token limits."

### Section 2: Controlled Overlap & Boundary Context Demonstration (1.5 min)
* "Show `src/token_chunker.py` and run `python token_chunking_demo.py`."
* "Show the boundary cut demonstration: When overlap is 0, a critical policy rule is sliced in half between Chunk #0 and Chunk #1, making retrieval fail. When we add a 60-token overlap, Chunk #1 contains the entire policy intact."

### Section 3: Parameter Justification & Trade-Offs (1 min)
* "We chose 400 tokens per chunk with 60 tokens (15%) overlap for OpenAI `text-embedding-3-small` and `GPT-4o-mini`."
* "Explain the cost trade-off: 15% overlap only adds ~9-15% token redundancy in storage, while preventing 100% of boundary context loss."

### Section 4: Follow-up Question: Chunk Size, Top-$k$, & Context Limits (45s)
* "How does chunk size interact with top-$k$ retrieval and the context window?"
* "Answer: They are a joint budget decision. $\text{Total Retrieval Tokens} = k \times \text{chunk\_size}$. Larger chunks mean fewer ($k$) can be retrieved into the context window, while smaller chunks allow retrieving more diverse sources ($k=6-8$) but risk fragmenting context. Our 400-token chunks with $k=4$ provide the perfect balance of semantic depth and prompt budget."
