# Chunking Strategy Justification (Task 4)

## Comparison
We evaluated two chunking strategies on our `customer_policy.txt` corpus:
1. **Fixed-Size with Overlap** (200 characters, 50 character overlap)
2. **Paragraph Chunking** (Splitting by `\n\n`)

## Justification for Chosen Strategy
For our RAG application and the `customer_policy.txt` corpus, **Paragraph Chunking** is the superior strategy.

**Why it fits our document structure:**
The policy document is naturally organized into distinct paragraphs, where each paragraph encapsulates a single, coherent policy rule or concept (e.g., Return Policy, Refund Timeline). 

**Why it fits our retrieval needs:**
1. **Semantic Completeness:** Fixed-size chunking blindly cuts sentences in half, leaving unresolved thoughts that confuse the LLM during generation. Paragraph chunking ensures boundaries align with natural thought completion.
2. **Precision:** By keeping an entire policy section intact, the embedding model can generate a much more accurate vector representation of that specific policy, meaning our vector search will retrieve the exact rule the user asked about without hallucinated half-sentences.
