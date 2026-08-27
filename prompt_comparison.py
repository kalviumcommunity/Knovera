import os
import json
import logging
from dotenv import load_dotenv
from openai import OpenAI, AuthenticationError, RateLimitError, APIError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("outputs/prompt_comparison.log"),
        logging.StreamHandler()
    ]
)

# Task 2: System message defining role, scope, constraints, and fallback behavior
SYSTEM_MESSAGE = (
    "You are Knovera, an internal support assistant for staff documentation. "
    "Your scope is to provide accurate, factual information regarding internal company policies. "
    "Answer concisely in a maximum of 2 sentences using a professional tone. "
    "If you are unsure or the information is not provided in context, state: "
    "'I do not have enough information to answer this question.'"
)

# Task 3: Compare prompt variations (Vague vs. Clear/Constrained)
PROMPT_VARIATIONS = [
    {
        "id": "Variation 1 (Vague Prompt)",
        "user_prompt": "Explain our refund policy.",
        "description": "Ambiguous query with no constraints on format, length, or specifics."
    },
    {
        "id": "Variation 2 (Clear & Constrained Prompt)",
        "user_prompt": "In one concise sentence, state the standard refund window in days and key eligibility criteria.",
        "description": "Specific query defining the exact task, required detail, and length constraint."
    },
    {
        "id": "Variation 3 (Explicit Format Constraint)",
        "user_prompt": "State our refund window. Reply ONLY with a valid JSON object matching this schema: {\"refund_window_days\": int, \"policy_summary\": string}.",
        "description": "Strict format constraint instructing the model to output parsed JSON for machine handling."
    }
]

MOCK_RESPONSES = {
    "Variation 1 (Vague Prompt)": (
        "Our standard refund policy allows customers to request a refund within 30 days of purchase for unused products. "
        "Please contact customer support with your original receipt and order ID to initiate the return process.",
        {"prompt_tokens": 58, "completion_tokens": 36, "total_tokens": 94}
    ),
    "Variation 2 (Clear & Constrained Prompt)": (
        "The standard refund window is 30 days from purchase for unused items with proof of purchase.",
        {"prompt_tokens": 64, "completion_tokens": 17, "total_tokens": 81}
    ),
    "Variation 3 (Explicit Format Constraint)": (
        '{\n  "refund_window_days": 30,\n  "policy_summary": "Refunds are eligible within 30 days of purchase for unused items with receipt."\n}',
        {"prompt_tokens": 76, "completion_tokens": 28, "total_tokens": 104}
    )
}

def run_prompt_comparison():
    load_dotenv()
    
    api_key = os.getenv("OPENROUTER_API_KEY", os.getenv("OPENAI_API_KEY"))
    base_url = os.getenv("OPENROUTER_BASE_URL", os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"))
    model_name = os.getenv("OPENROUTER_MODEL", os.getenv("OPENAI_MODEL", os.getenv("CHAT_MODEL", "gpt-3.5-turbo")))
    
    use_mock = not api_key or api_key == "your_openai_api_key_here"
    if use_mock:
        logging.warning("No valid API key found in .env. Running in offline comparison demo mode.")
        print("[NOTICE] Running in comparison demo mode (offline simulation). Set OPENROUTER_API_KEY in .env for live API calls.\n")
    else:
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        client = OpenAI(**client_kwargs)


    output_lines = []
    output_lines.append("=========================================================================")
    output_lines.append("                KNOVERA - PROMPT VARIATION COMPARISON                    ")
    output_lines.append("=========================================================================\n")
    output_lines.append(f"System Message Role & Constraints:\n{SYSTEM_MESSAGE}\n")
    output_lines.append("-------------------------------------------------------------------------\n")

    for idx, var in enumerate(PROMPT_VARIATIONS, 1):
        output_lines.append(f"--- Prompt {var['id']} ---")
        output_lines.append(f"Description: {var['description']}")
        output_lines.append(f"User Prompt: \"{var['user_prompt']}\"")
        
        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": var["user_prompt"]}
        ]
        
        try:
            if use_mock:
                logging.info(f"Generating mock output for {var['id']}...")
                reply, usage_dict = MOCK_RESPONSES.get(var['id'], ("Sample reply.", {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}))
                output_lines.append(f"Assistant Reply:\n{reply}")
                output_lines.append(f"Tokens Used: Prompt={usage_dict['prompt_tokens']}, Completion={usage_dict['completion_tokens']}, Total={usage_dict['total_tokens']}\n")
            else:
                logging.info(f"Executing call for {var['id']} using model '{model_name}'...")
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.3
                )
                
                reply = response.choices[0].message.content.strip()
                usage = response.usage
                
                output_lines.append(f"Assistant Reply:\n{reply}")
                output_lines.append(f"Tokens Used: Prompt={usage.prompt_tokens}, Completion={usage.completion_tokens}, Total={usage.total_tokens}\n")
            
        except AuthenticationError:
            err_msg = "[ERROR 401] Authentication failed. Check your API key."
            logging.error(err_msg)
            output_lines.append(f"Assistant Reply: {err_msg}\n")
        except RateLimitError:
            err_msg = "[ERROR 429] Rate limit exceeded or quota exhausted."
            logging.error(err_msg)
            output_lines.append(f"Assistant Reply: {err_msg}\n")
        except APIError as e:
            err_msg = f"[ERROR API] API call failed: {e}"
            logging.error(err_msg)
            output_lines.append(f"Assistant Reply: {err_msg}\n")
        except Exception as e:
            err_msg = f"[ERROR Unexpected] {e}"
            logging.error(err_msg)
            output_lines.append(f"Assistant Reply: {err_msg}\n")

    full_output = "\n".join(output_lines)
    print(full_output)

    # Save output to outputs directory
    os.makedirs("outputs", exist_ok=True)
    output_filepath = os.path.join("outputs", "prompt_comparison_output.txt")
    with open(output_filepath, "w", encoding="utf-8") as f:
        f.write(full_output)
    
    logging.info(f"Comparison results successfully written to {output_filepath}")

if __name__ == "__main__":
    run_prompt_comparison()
