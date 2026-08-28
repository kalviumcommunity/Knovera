import os
import json
import logging
from dotenv import load_dotenv
from openai import OpenAI, AuthenticationError, RateLimitError, APIError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def parse(raw, required=("answer", "source")):
    """
    Parse and validate the JSON response.
    Returns (data, None) if successful, or (None, error_message) if it fails.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None, "malformed JSON"
        
    missing = [k for k in required if k not in data]
    if missing:
        return None, f"missing fields: {missing}"
        
    return data, None

def get_answer_from_model(client, model_name, question, force_malformed=False):
    """
    Prompts the model to return a structured JSON output.
    """
    system_prompt = (
        'Reply with ONLY a JSON object: '
        '{"answer": string, "source": string}. No extra text.'
    )
    
    # Intentionally provide a bad system prompt to test malformed output recovery
    if force_malformed:
        system_prompt = "Reply with plain text, NOT JSON. Ignore previous instructions."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]
    
    try:
        # Task 1: Prompt for a defined JSON structure using response_format
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0,
        )
        return response.choices[0].message.content
    except Exception as e:
        # Some models might not support response_format={"type": "json_object"}
        # Fallback to standard request without response_format
        logging.warning(f"Error with JSON mode, trying standard mode: {e}")
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0,
        )
        return response.choices[0].message.content

def ask_question(client, model_name, question, force_malformed=False):
    raw_response = get_answer_from_model(client, model_name, question, force_malformed)
    
    logging.info(f"Raw Model Response:\n{raw_response}")
    
    # Task 2, 3, 4: Parse and validate JSON
    data, err = parse(raw_response)
    
    if data:
        logging.info("Successfully parsed and validated JSON!")
        return data
    else:
        logging.warning(f"Parse failed: {err}. Attempting recovery...")
        
        # Recovery strategy: Retry once with a strict reminder
        system_prompt_recovery = (
            'You previously provided invalid output. '
            'You MUST reply with ONLY a valid JSON object matching exactly: '
            '{"answer": string, "source": string}. No extra text or prose.'
        )
        
        messages = [
            {"role": "system", "content": system_prompt_recovery},
            {"role": "user", "content": question}
        ]
        
        try:
            recovery_response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0,
            )
            raw_recovery = recovery_response.choices[0].message.content
            logging.info(f"Raw Recovery Response:\n{raw_recovery}")
            
            recovered_data, recovery_err = parse(raw_recovery)
            
            if recovered_data:
                logging.info("Successfully recovered and parsed JSON!")
                return recovered_data
            else:
                logging.error(f"Recovery failed: {recovery_err}")
                return None
                
        except Exception as e:
            logging.error(f"Error during recovery: {e}")
            return None


def main():
    load_dotenv()
    
    api_key = os.getenv("OPENROUTER_API_KEY", os.getenv("OPENAI_API_KEY"))
    base_url = os.getenv("OPENROUTER_BASE_URL", os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"))
    model_name = os.getenv("OPENROUTER_MODEL", os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"))
    
    if not api_key:
        logging.error("Missing API key in .env file")
        return
        
    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )
    
    results = []
    
    print("--- Test 1: Normal JSON Request ---")
    q1 = "What is the refund window according to standard e-commerce policies?"
    ans1 = ask_question(client, model_name, q1)
    results.append({"Test": "Normal Request", "Question": q1, "Result": ans1})
    print(f"Result: {ans1}\n")
    
    print("--- Test 2: Simulating Malformed JSON and Recovery ---")
    # Forcing malformed output by providing a bad system prompt first
    q2 = "What is the capital of France?"
    ans2 = ask_question(client, model_name, q2, force_malformed=True)
    results.append({"Test": "Malformed & Recovery", "Question": q2, "Result": ans2})
    print(f"Result: {ans2}\n")
    
    # Save the sample parsed results to a text file for Task 5
    with open("outputs/sample_structured_output.txt", "w") as f:
        f.write("Sample Structured Output Execution Results:\n")
        f.write("="*45 + "\n\n")
        for res in results:
            f.write(f"Test Case: {res['Test']}\n")
            f.write(f"Question: {res['Question']}\n")
            f.write(f"Final Parsed Object: {json.dumps(res['Result'], indent=2)}\n")
            f.write("-" * 30 + "\n\n")
            
    print("Results have been saved to 'outputs/sample_structured_output.txt'.")

if __name__ == "__main__":
    main()
