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
