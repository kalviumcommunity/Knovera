import sys
from templates.prompts import QA_PROMPT_TEMPLATE, render_prompt

def batch_process_questions(questions: list[str], shared_context: str) -> list[str]:
    """
    Task 3: Batch/CLI feature reusing the SAME template structure.
    """
    prompts = []
    for q in questions:
        # Task 2: Inject dynamic values at runtime in a loop.
        prompt = render_prompt(
            QA_PROMPT_TEMPLATE,
            context=shared_context,
            question=q
        )
        prompts.append(prompt)
    
    return prompts

if __name__ == "__main__":
    qs = ["What is the IP?", "What is the port?"]
    ctx = "Server configuration: IP is 192.168.1.5, port is 8080."
    rendered_prompts = batch_process_questions(qs, ctx)
    
    print("CLI Feature Generated Prompts:\n")
    for i, p in enumerate(rendered_prompts, 1):
        print(f"--- Request {i} ---")
        print(p)
        print()
