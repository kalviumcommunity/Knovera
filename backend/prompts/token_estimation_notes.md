# Tokens, Tokenization & Cost Estimation Analysis (Assignment 3.14)

## 1. Concepts & Definitions

### What is a Token vs. Word or Character?
- **Character**: The smallest written unit of text (letters, numbers, punctuation, spaces, emojis). In raw text string representation, 1 character = 1 byte (or multi-byte in UTF-8).
- **Word**: A sequence of characters separated by spaces or punctuation (e.g., "refund", "internationalization").
- **Token**: The atomic chunk of text processed by a Large Language Model (LLM) using sub-word Byte-Pair Encoding (BPE) algorithms (such as `cl100k_base` used by GPT-3.5/GPT-4 or `o200k_base` by GPT-4o).
  - In standard English text, 1 token is roughly **3/4 of a word** (~4 characters).
  - Common words like `"refund"` are 1 token.
  - Less common, long, or complex words like `"refundable"` or `"hyperparameterization"` are split into multiple tokens (`"refund"` + `"able"` or `"hyper"` + `"parameter"` + `"ization"`).
  - Punctuation, spaces, numbers, code formatting (`{`, `}`, `"`, `\n`), and emojis also count as distinct tokens.

### Why Tokens Decide Cost and Context Limits
1. **Cost**: Model providers bill per token (typically quoted per 1,000 or 1,000,000 tokens). Crucially, **input tokens (prompts + retrieved context)** and **output tokens (generated completions)** are priced differently, with output tokens generally being **2x to 4x more expensive**.
2. **Context Limits**: Every model has a fixed maximum context window (e.g., 4,096 tokens for base models, 128k for GPT-4o). The **System Prompt + Retrieved Document Chunks + Message History + Generated Answer** must fit within this limit. Exceeding the window results in truncated responses or API rejection errors (`context_length_exceeded`).

---

## 2. Project Samples Token Measurement (Task 1 & Task 2)

Measured using `tiktoken` (`cl100k_base` tokenizer):

| Sample Label | Text Description | Character Count | Word Count | Token Count | Chars / Token | Tokens / Word |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Sample 1 (Short Query)** | User prompt asking standard refund window. | 69 | 10 | **11** | 6.27 | 1.10 |
| **Sample 2 (Medium Section)** | Paragraph extract of Customer Satisfaction Policy. | 370 | 58 | **69** | 5.36 | 1.19 |
| **Sample 3 (Long SOP Document)** | Full internal SOP document context chunk. | 1,530 | 214 | **288** | 5.31 | 1.35 |

---

## 3. Cost Estimation Breakdown (Task 3)

### Formula
$$\text{Cost} = \left(\frac{\text{Input Tokens}}{1000} \times \text{Input Rate}\right) + \left(\frac{\text{Output Tokens}}{1000} \times \text{Output Rate}\right)$$

### Single RAG Query Execution
- **System Prompt**: 71 tokens
- **Retrieved Document Context (Sample 3)**: 288 tokens
- **User Question (Sample 1)**: 11 tokens
- **Total Input Tokens**: **370 tokens**
- **Expected Output Completion**: **45 tokens**

| Model Provider | Input Rate (/1K) | Output Rate (/1K) | Input Cost | Output Cost | Total Cost / Query | Cost per 1,000 Calls |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **GPT-3.5-Turbo** | $0.00050 | $0.00150 | $0.000185 | $0.000068 | **$0.000253** | **$0.253** |
| **GPT-4o-Mini** | $0.00015 | $0.00060 | $0.000055 | $0.000027 | **$0.000082** | **$0.082** |
| **GPT-4o** | $0.00250 | $0.01000 | $0.000925 | $0.000450 | **$0.001375** | **$1.375** |

### Corpus Scale Analysis (4,000 Documents)

1. **One-Time Document Embedding Ingestion**:
   - 4,000 documents $\times$ 288 tokens/doc = **1,152,000 total tokens**.
   - Using `text-embedding-3-small` ($0.00002 per 1K tokens):
     $$\text{Embedding Cost} = \frac{1,152,000}{1000} \times 0.00002 = \mathbf{\$0.0230}$$

2. **Monthly Operations (10,000 Queries with Top-3 Chunk Retrieval)**:
   - Input Tokens per Query: 71 (System) + (3 $\times$ 288 Context) + 11 (User) = **946 tokens**.
   - Total Monthly Input: $10,000 \times 946 = \mathbf{9,460,000 \text{ tokens}}$.
   - Total Monthly Output: $10,000 \times 45 = \mathbf{450,000 \text{ tokens}}$.
   - **Monthly Cost (GPT-3.5-Turbo)**: $\$4.73 \text{ (Input)} + \$0.68 \text{ (Output)} = \mathbf{\$5.41 / \text{month}}$.
   - **Monthly Cost (GPT-4o-Mini)**: $\$1.42 \text{ (Input)} + \$0.27 \text{ (Output)} = \mathbf{\$1.69 / \text{month}}$.

---

## 4. Demonstrating the Length–Token Relationship (Task 4)

Text length and token counts track together, but are **non-linear** due to sub-word tokenization mechanics:

| Category | Sample Text Snippet | Chars | Words | Tokens | Chars/Tok | Tok/Word | Tokenization Behavior |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **1. Standard English** | `"The standard refund window is 30 calendar days..."` | 92 | 16 | **18** | 5.11 | 1.12 | Standard English prose (~4 chars/token). Most common words are 1 token. |
| **2. Technical Jargon** | `"unrefundable hyperparameterization internationalization..."` | 84 | 5 | **11** | 7.64 | 2.20 | Long or uncommon words split into multiple sub-word tokens (`un`+`refundable`). |
| **3. Code & JSON Payload** | `{\n  "refund_window_days": 30,\n  "status": "APPROVED"}` | 112 | 11 | **42** | 2.67 | 3.82 | Brackets, quotes, colons, spaces, and newlines each consume individual tokens. |
| **4. Multilingual & Emojis** | `"Politique... 返金ポリシー... 🚀 Priority Support! 💰"` | 89 | 13 | **37** | 2.41 | 2.85 | Non-ASCII alphabets & Unicode emojis require multiple byte-level tokens per character. |

---

## 5. Video Explanation Outline (Walkthrough Guide)

When recording your 3-5 minute video explanation:
1. **Introduction & Token Definition**: Explain that models do not read words or characters directly; text is tokenized using Byte-Pair Encoding (BPE). 1 token $\approx$ 0.75 words in English.
2. **Cost & Context Window Impact**:
   - Cost is calculated per token; input tokens (context) and output tokens (answers) have distinct rates (output is pricier).
   - Context limits cap total allowable tokens (prompt + context + history + output).
3. **Code Walkthrough (`token_estimation.py`)**:
   - Show how `tiktoken.get_encoding("cl100k_base")` is initialized.
   - Walk through token counting across the 3 project samples (short query, medium policy paragraph, long SOP chunk).
   - Show the cost calculation functions and rates.
4. **Length-Token Non-Linearity**:
   - Demonstrate why code/JSON and multilingual/emoji samples produce higher token counts per word than standard English.
5. **Corpus Scale Answer**:
   - Explain how embedding 4,000 documents multiplied by daily query retrieval (e.g. top-k=3) scales token usage into millions per month, emphasizing why token measurement up-front prevents unexpected API bills.
