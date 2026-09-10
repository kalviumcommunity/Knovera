# Task 4: Separate templates from logic

# Task 1: Define a template with named placeholders
QA_PROMPT_TEMPLATE = """You are an expert technical assistant.
Please answer the user's question using only the provided context.

Context:
{context}

Question:
{question}

Answer:"""

def render_prompt(template_str: str, **kwargs) -> str:
    """
    Task 1: Render function that fills placeholders safely.
    Raises a KeyError if a required placeholder is missing.
    """
    try:
        return template_str.format(**kwargs)
    except KeyError as e:
        raise ValueError(f"Missing required template variable: {e}")

GROUNDED_RAG_PROMPT_TEMPLATE = """You are a grounded assistant for Knovera.
Answer the question using ONLY the provided context below.
If the answer is not in the context, or if the context is insufficient, explicitly say:
"I don't have enough information in the provided context."
Do not extrapolate or speculate beyond the provided evidence.
When possible, cite sources using the citation markers like [1] or [2].

Context:
{context}

Question:
{question}

Answer:"""

