from templates.prompts import QA_PROMPT_TEMPLATE, render_prompt

def handle_chat_request(user_question: str, kb_context: str) -> str:
    """
    Task 3: Chat feature reusing the template.
    Task 2: Inject dynamic values at runtime.
    """
    # In a real app, this would query a vector DB for context.
    # We dynamically render the prompt before sending to an LLM.
    final_prompt = render_prompt(
        QA_PROMPT_TEMPLATE,
        context=kb_context,
        question=user_question
    )
    
    # return the prompt (simulating sending it to an LLM)
    return final_prompt

if __name__ == "__main__":
    prompt = handle_chat_request("How do I reset my password?", "Go to settings > security > reset.")
    print("Chat Feature Generated Prompt:\n")
    print(prompt)
