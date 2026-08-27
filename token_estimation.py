import os
import sys
import logging
import tiktoken

# Reconfigure stdout to UTF-8 for Windows console support
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Configure logging
os.makedirs("outputs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("outputs/token_estimation.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)

# System Prompt Template (from prompts/system_prompt.txt)
SYSTEM_PROMPT = (
    "You are Knovera, an internal support assistant for staff documentation. "
    "Your scope is to provide accurate, factual information regarding internal company policies. "
    "Answer concisely in a maximum of 2 sentences using a professional tone. "
    "If you are unsure or the information is not provided in context, state: "
    "'I do not have enough information to answer this question.'"
)

# Sample 1: Short Query
SAMPLE_SHORT = "What is our standard customer refund window and eligibility criteria?"

# Sample 2: Medium Paragraph (Policy Section)
SAMPLE_MEDIUM = (
    "Knovera Customer Satisfaction & Refund Policy: Standard refund requests must be "
    "submitted within 30 calendar days of the original purchase date. Items must be unused, "
    "in original packaging, and accompanied by a valid sales receipt or order reference ID. "
    "Returns submitted after 30 days or without proof of purchase are subject to store credit "
    "only upon manager approval."
)

# Sample 3: Long Document (Full SOP Document / Context Chunk)
SAMPLE_LONG = (
    "Knovera Standard Operating Procedure (SOP-FIN-2026-04): Customer Returns and Refund Processing\n\n"
    "1. Overview & Purpose\n"
    "This document outlines the standard procedures for handling customer return requests, refund evaluations, "
    "and accounting reconciliations within the Knovera enterprise ecosystem. All support staff must adhere strictly "
    "to these guidelines to maintain financial compliance and service consistency.\n\n"
    "2. Standard Return Window & Eligibility\n"
    "Customers are eligible for a full monetary refund if a request is initiated within 30 calendar days of delivery "
    "or purchase. To qualify, merchandise must be in original condition, unaltered, and accompanied by proof of purchase "
    "(invoice ID or sales receipt). Customized orders, downloadable digital software licenses, and opened consumable goods "
    "are non-refundable unless verified defective by technical support.\n\n"
    "3. Verification & Processing Workflow\n"
    "Upon receiving a return request, support agents must verify the order status in the Knovera CRM system. "
    "Once verified, issue a Return Merchandise Authorization (RMA) tracking number. Refunds will be credited back "
    "to the original payment method within 5 to 7 business days following physical receipt and inspection of merchandise.\n\n"
    "4. Exception Handling & Managerial Escalations\n"
    "For requests made between 31 and 60 days, store credit may be issued at managerial discretion. "
    "Any refund request exceeding $1,000 USD or requesting wire transfer reimbursement requires dual sign-off from "
    "the Operations Lead and Finance Director."
)

# Task 4 Demonstration Samples
DEMO_SAMPLES = [
    {
        "category": "1. Standard English Text",
        "text": "The standard refund window is 30 calendar days from the date of purchase with valid receipt.",
        "note": "Typical English prose (~4 chars per token, ~0.75-0.8 words per token)."
    },
    {
        "category": "2. Technical Jargon / Long Words",
        "text": "unrefundable hyperparameterization internationalization microservices infrastructure",
        "note": "Long or compound words split into sub-word tokens (e.g. 'un' + 'refundable')."
    },
    {
        "category": "3. Code & Formatted JSON Payload",
        "text": '{\n  "refund_window_days": 30,\n  "status": "APPROVED",\n  "metadata": {"user_id": "usr_99812", "amount": 149.99}\n}',
        "note": "Syntax symbols ({}, quotes, colons, newlines, indents) increase token count per character."
    },
    {
        "category": "4. Multilingual & Special Symbols",
        "text": "Politique de remboursement: 30 jours. 返金ポリシー: 30日. 🚀 Priority Support: 100% Guaranteed! 💰",
        "note": "Non-English alphabets and emojis require multiple byte-level tokens per character."
    }
]

# Price Rates per 1,000 Tokens (USD)
PRICING_MODELS = {
    "GPT-3.5-Turbo": {"input": 0.0005, "output": 0.0015, "embedding": 0.00010},
    "GPT-4o-Mini":   {"input": 0.00015, "output": 0.00060, "embedding": 0.00002},
    "GPT-4o":        {"input": 0.0025,  "output": 0.0100,  "embedding": 0.00002},
    "Ling-3.0-Tiny": {"input": 0.0, "output": 0.0, "embedding": 0.0}
}


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> dict:
    """
    Counts tokens for a given text using tiktoken tokenizer.
    Returns details including token count, character count, word count, and token ratios.
    """
    try:
        enc = tiktoken.get_encoding(encoding_name)
    except Exception as e:
        logging.warning(f"Encoding {encoding_name} not found, falling back to cl100k_base: {e}")
        enc = tiktoken.get_encoding("cl100k_base")

    token_ids = enc.encode(text)
    token_count = len(token_ids)
    char_count = len(text)
    word_count = len(text.split())

    chars_per_token = char_count / token_count if token_count > 0 else 0.0
    tokens_per_word = token_count / word_count if word_count > 0 else 0.0

    return {
        "text": text,
        "token_count": token_count,
        "char_count": char_count,
        "word_count": word_count,
        "chars_per_token": chars_per_token,
        "tokens_per_word": tokens_per_word,
        "token_ids": token_ids
    }


def calculate_cost(input_tokens: int, output_tokens: int, rate_input_per_1k: float, rate_output_per_1k: float) -> dict:
    """
    Calculates cost based on input and output token counts and their respective pricing rates.
    """
    input_cost = (input_tokens / 1000.0) * rate_input_per_1k
    output_cost = (output_tokens / 1000.0) * rate_output_per_1k
    total_cost = input_cost + output_cost

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost
    }


def run_token_estimation():
    logging.info("Starting Tokenization and Cost Estimation Analysis...")

    output_lines = []
    output_lines.append("=========================================================================")
    output_lines.append("        KNOVERA - TOKENS, TOKENIZATION & COST ESTIMATION REPORT         ")
    output_lines.append("=========================================================================\n")

    # -------------------------------------------------------------------------
    # Task 1 & Task 2: Token Counting for Three Project Samples
    # -------------------------------------------------------------------------
    output_lines.append("-------------------------------------------------------------------------")
    output_lines.append("TASK 1 & TASK 2: TOKEN COUNTS FOR THREE PROJECT SAMPLES")
    output_lines.append("-------------------------------------------------------------------------")
    
    samples = [
        ("Sample 1 (Short User Query)", SAMPLE_SHORT),
        ("Sample 2 (Medium Policy Section)", SAMPLE_MEDIUM),
        ("Sample 3 (Long SOP Document / Context Chunk)", SAMPLE_LONG)
    ]

    sample_metrics = []
    for label, text in samples:
        res = count_tokens(text)
        sample_metrics.append((label, res))
        
        output_lines.append(f"\n--- {label} ---")
        output_lines.append(f"Text Snippet: \"{text[:120]}{'...' if len(text) > 120 else ''}\"")
        output_lines.append(f"  • Character Count : {res['char_count']} chars")
        output_lines.append(f"  • Word Count      : {res['word_count']} words")
        output_lines.append(f"  • Token Count     : {res['token_count']} tokens")
        output_lines.append(f"  • Chars / Token   : {res['chars_per_token']:.2f}")
        output_lines.append(f"  • Tokens / Word   : {res['tokens_per_word']:.2f}")

    # Summary Table for 3 Samples
    output_lines.append("\nSummary Table - Project Samples:")
    output_lines.append(f"{'Sample Label':<42} | {'Chars':<6} | {'Words':<6} | {'Tokens':<6} | {'Chars/Tok':<9}")
    output_lines.append("-" * 78)
    for label, res in sample_metrics:
        output_lines.append(
            f"{label:<42} | {res['char_count']:<6} | {res['word_count']:<6} | {res['token_count']:<6} | {res['chars_per_token']:<9.2f}"
        )

    # -------------------------------------------------------------------------
    # Task 3: Cost Estimation from Token Counts
    # -------------------------------------------------------------------------
    output_lines.append("\n-------------------------------------------------------------------------")
    output_lines.append("TASK 3: COST ESTIMATION (INPUT VS OUTPUT BILLING)")
    output_lines.append("-------------------------------------------------------------------------")

    # Scenario 1: Single RAG Call
    # Input = System Prompt + Long Document (Context) + Short Question
    sys_res = count_tokens(SYSTEM_PROMPT)
    doc_res = sample_metrics[2][1]  # Sample 3 Long SOP
    query_res = sample_metrics[0][1] # Sample 1 Short Question
    
    rag_input_tokens = sys_res['token_count'] + doc_res['token_count'] + query_res['token_count']
    estimated_output_tokens = 45  # Typical concise assistant reply

    output_lines.append("\n[Scenario A: Single RAG Query Execution]")
    output_lines.append(f"  • System Prompt Tokens  : {sys_res['token_count']}")
    output_lines.append(f"  • Context Chunk Tokens  : {doc_res['token_count']}")
    output_lines.append(f"  • User Query Tokens     : {query_res['token_count']}")
    output_lines.append(f"  • Total Input Tokens    : {rag_input_tokens}")
    output_lines.append(f"  • Expected Output Tokens: {estimated_output_tokens}")

    output_lines.append("\nCost Breakdown Across Model Providers:")
    for model_name, rates in PRICING_MODELS.items():
        cost_info = calculate_cost(
            input_tokens=rag_input_tokens,
            output_tokens=estimated_output_tokens,
            rate_input_per_1k=rates["input"],
            rate_output_per_1k=rates["output"]
        )
        output_lines.append(f"\n  Model: {model_name}")
        output_lines.append(f"    - Input Rate  : ${rates['input']:.5f} / 1K tokens  | Input Cost : ${cost_info['input_cost']:.6f}")
        output_lines.append(f"    - Output Rate : ${rates['output']:.5f} / 1K tokens  | Output Cost: ${cost_info['output_cost']:.6f}")
        output_lines.append(f"    - TOTAL COST PER CALL: ${cost_info['total_cost']:.6f} (~${cost_info['total_cost']*1000:.3f} per 1K calls)")

    # Scenario 2: Corpus Scale Estimation (4,000 Documents)
    output_lines.append("\n[Scenario B: Corpus Scale & Retrieval Operations (4,000 Documents)]")
    num_docs = 4000
    avg_doc_tokens = doc_res['token_count'] # ~320 tokens per document chunk
    total_corpus_tokens = num_docs * avg_doc_tokens
    
    output_lines.append(f"  1. One-time Corpus Ingestion & Embedding Cost:")
    output_lines.append(f"     - Total Corpus Documents  : {num_docs:,}")
    output_lines.append(f"     - Average Tokens / Doc    : {avg_doc_tokens} tokens")
    output_lines.append(f"     - Total Corpus Tokens     : {total_corpus_tokens:,} tokens")
    
    emb_rate = PRICING_MODELS["GPT-4o-Mini"]["embedding"] # text-embedding-3-small rate $0.00002 / 1K
    corpus_emb_cost = (total_corpus_tokens / 1000.0) * emb_rate
    output_lines.append(f"     - Embedding Cost ($0.00002/1K): ${corpus_emb_cost:.4f}")

    output_lines.append(f"\n  2. Monthly Operational Query Scaling (10,000 Queries/Month with Top-3 Chunk Retrieval):")
    queries_per_month = 10000
    retrieved_chunks = 3
    monthly_input_tokens_per_query = sys_res['token_count'] + (retrieved_chunks * avg_doc_tokens) + query_res['token_count']
    total_monthly_input_tokens = queries_per_month * monthly_input_tokens_per_query
    total_monthly_output_tokens = queries_per_month * estimated_output_tokens

    cost_gpt35 = calculate_cost(total_monthly_input_tokens, total_monthly_output_tokens, PRICING_MODELS["GPT-3.5-Turbo"]["input"], PRICING_MODELS["GPT-3.5-Turbo"]["output"])
    cost_gpt4o_mini = calculate_cost(total_monthly_input_tokens, total_monthly_output_tokens, PRICING_MODELS["GPT-4o-Mini"]["input"], PRICING_MODELS["GPT-4o-Mini"]["output"])
    cost_ling = calculate_cost(total_monthly_input_tokens, total_monthly_output_tokens, PRICING_MODELS["Ling-3.0-Tiny"]["input"], PRICING_MODELS["Ling-3.0-Tiny"]["output"])

    output_lines.append(f"     - Tokens per Query (Input)  : {monthly_input_tokens_per_query} tokens (3 chunks retrieved)")
    output_lines.append(f"     - Total Monthly Input Tokens: {total_monthly_input_tokens:,} tokens")
    output_lines.append(f"     - Total Monthly Output Tokens: {total_monthly_output_tokens:,} tokens")
    output_lines.append(f"     - Estimated Monthly Cost (GPT-3.5-Turbo) : ${cost_gpt35['total_cost']:.2f}")
    output_lines.append(f"     - Estimated Monthly Cost (GPT-4o-Mini)   : ${cost_gpt4o_mini['total_cost']:.2f}")
    output_lines.append(f"     - Estimated Monthly Cost (Ling-3.0-Tiny) : ${cost_ling['total_cost']:.2f}")

    # -------------------------------------------------------------------------
    # Task 4: Length–Token Relationship Demonstration
    # -------------------------------------------------------------------------
    output_lines.append("\n-------------------------------------------------------------------------")
    output_lines.append("TASK 4: LENGTH–TOKEN RELATIONSHIP & NON-LINEARITY DEMONSTRATION")
    output_lines.append("-------------------------------------------------------------------------")
    output_lines.append("Demonstrating that character length and word count track with token count,")
    output_lines.append("but are NOT strictly proportional due to sub-word BPE tokenization:\n")

    output_lines.append(f"{'Category':<35} | {'Chars':<5} | {'Words':<5} | {'Tokens':<6} | {'Chars/Tok':<9} | {'Tok/Word':<8}")
    output_lines.append("-" * 80)

    for item in DEMO_SAMPLES:
        res = count_tokens(item["text"])
        output_lines.append(
            f"{item['category']:<35} | {res['char_count']:<5} | {res['word_count']:<5} | {res['token_count']:<6} | {res['chars_per_token']:<9.2f} | {res['tokens_per_word']:<8.2f}"
        )

    output_lines.append("\nDetailed Insights & Analysis:")
    for item in DEMO_SAMPLES:
        res = count_tokens(item["text"])
        output_lines.append(f"\n[{item['category']}]")
        output_lines.append(f"  Sample Text : \"{item['text']}\"")
        output_lines.append(f"  Note        : {item['note']}")
        output_lines.append(f"  Metrics     : {res['char_count']} chars, {res['word_count']} words => {res['token_count']} tokens.")
        output_lines.append(f"  Token IDs   : {res['token_ids']}")

    output_lines.append("\nKey Takeaways on Non-Linear Tokenization:")
    output_lines.append("  1. Standard English prose averages ~4 characters per token (~0.75 words per token).")
    output_lines.append("  2. Long, technical, or un-common words split into multiple sub-word tokens (e.g. 'hyperparameterization').")
    output_lines.append("  3. Code and JSON formatting contain spaces, quotes, colons, and brackets that increase token density.")
    output_lines.append("  4. Multilingual text and Unicode emojis take multiple byte tokens per character, making them significantly more expensive.")

    full_output = "\n".join(output_lines)
    print(full_output)

    # Save outputs
    output_filepath = os.path.join("outputs", "token_estimation_output.txt")
    with open(output_filepath, "w", encoding="utf-8") as f:
        f.write(full_output)
        
    logging.info(f"Token estimation analysis successfully written to {output_filepath}")

if __name__ == "__main__":
    run_token_estimation()
