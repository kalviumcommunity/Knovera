from feature_chat import handle_chat_request
from feature_cli import batch_process_questions

def main():
    print("========================================")
    print("TASK 5: EXAMPLE RENDERS")
    print("========================================\n")
    
    print("1. Chat Feature Render:\n")
    chat_prompt = handle_chat_request(
        "Why is the database slow?",
        "Database is experiencing high lock contention on the user table."
    )
    print(chat_prompt)
    print("\n----------------------------------------\n")
    
    print("2. Batch CLI Feature Render:\n")
    batch_prompts = batch_process_questions(
        ["How do I start the server?", "How do I stop it?"],
        "Commands: 'start.sh' to boot, 'stop.sh' to halt."
    )
    for i, p in enumerate(batch_prompts, 1):
        print(f"[Batch Item {i}]")
        print(p)
        print()
        
if __name__ == "__main__":
    main()
