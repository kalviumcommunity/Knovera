# RAG Assistant Parameter Guidelines

For a Retrieval-Augmented Generation (RAG) assistant, the primary goal is **factual accuracy, consistency, and cost-efficiency**. An assistant that invents information (hallucinates) or rambles on indefinitely is unhelpful and expensive.

The following model parameter settings are highly recommended for grounded RAG tasks:

## 1. Temperature (`temperature`)
**Recommended Value: `0.0` to `0.2`**
- **Why?** Temperature controls the randomness of the model's output. A high temperature (e.g., `1.0`) makes the model more creative, which is great for writing poetry but dangerous for factual queries because it encourages hallucination.
- A low temperature ensures the model's responses are deterministic, stable, and firmly grounded in the retrieved context.

## 2. Max Tokens (`max_tokens`)
**Recommended Value: `300` to `500` (depending on expected response length)**
- **Why?** `max_tokens` puts a hard cap on the length of the response. Since LLM APIs typically bill by the output token, this is your primary defense against runaway costs caused by a model getting stuck in a loop or writing an essay when a sentence would suffice.
- It also forces concise answers, which is usually preferred for internal documentation assistants.

## 3. Stop Sequences (`stop`)
**Recommended Value: `["\n\nUser:", "User:"]` or specific context markers**
- **Why?** Stop sequences tell the model to halt generation immediately when it predicts a specific string. 
- In RAG architectures, models sometimes hallucinate a simulated back-and-forth conversation if they aren't stopped correctly. A stop sequence ensures the assistant provides its answer and immediately yields the floor back to the user without rambling or hallucinating a follow-up query.
