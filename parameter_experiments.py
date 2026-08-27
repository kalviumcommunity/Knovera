import os
import logging
from dotenv import load_dotenv
from openai import OpenAI

# Configure logging
os.makedirs("outputs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("outputs/parameter_experiments.log"),
        logging.StreamHandler()
    ]
)

def run_experiments():
    load_dotenv()
    
    api_key = os.getenv("OPENROUTER_API_KEY", os.getenv("OPENAI_API_KEY"))
    base_url = os.getenv("OPENROUTER_BASE_URL", os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"))
    model_name = os.getenv("OPENROUTER_MODEL", os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"))
    
    if not api_key:
        logging.error("Missing API key in .env")
        return
        
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        default_headers={
            "HTTP-Referer": "http://localhost:5173",
            "X-Title": "Knovera"
        }
    )
    
    system_prompt = "You are Knovera, a highly knowledgeable support assistant. Provide detailed and creative responses when possible, but stay on topic."
    user_prompt = "Can you explain the history of refund policies in the retail industry in exactly 3 long paragraphs?"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append("KNOVERA - MODEL PARAMETER EXPERIMENTS")
    output_lines.append("=" * 80)
    output_lines.append(f"Model used: {model_name}\n")
    
    # ---------------------------------------------------------
    # Experiment 1: Temperature
    # ---------------------------------------------------------
    output_lines.append("--- EXPERIMENT 1: TEMPERATURE (0.0 vs 1.0) ---")
    output_lines.append("Testing how randomness affects output consistency and creativity.\n")
    
    for t in [0.0, 1.0]:
        logging.info(f"Running Temperature test with t={t}...")
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=t
            )
            content = response.choices[0].message.content
            output_lines.append(f"[Temperature = {t}]")
            output_lines.append(f"Result (first 250 chars):\n{content[:250]}...\n")
        except Exception as e:
            output_lines.append(f"[Temperature = {t}] Error: {e}\n")
    
    # ---------------------------------------------------------
    # Experiment 2: Max Tokens
    # ---------------------------------------------------------
    output_lines.append("\n--- EXPERIMENT 2: MAX TOKENS (20 vs 300) ---")
    output_lines.append("Testing how max_tokens caps output length to protect against runaway costs.\n")
    
    for tokens in [20, 300]:
        logging.info(f"Running Max Tokens test with limit={tokens}...")
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.5,
                max_tokens=tokens
            )
            content = response.choices[0].message.content
            usage = response.usage
            output_lines.append(f"[Max Tokens = {tokens}]")
            output_lines.append(f"Tokens Used: Prompt={usage.prompt_tokens}, Completion={usage.completion_tokens}")
            output_lines.append(f"Result:\n{content}\n")
        except Exception as e:
            output_lines.append(f"[Max Tokens = {tokens}] Error: {e}\n")

    # ---------------------------------------------------------
    # Experiment 3: Stop Sequence
    # ---------------------------------------------------------
    output_lines.append("\n--- EXPERIMENT 3: STOP SEQUENCE ---")
    output_lines.append("Testing early stopping to prevent the model from rambling past a specific point.\n")
    
    stop_sequence = ["refund"]
    logging.info(f"Running Stop Sequence test with stop={stop_sequence}...")
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.5,
            stop=stop_sequence
        )
        content = response.choices[0].message.content
        output_lines.append(f"[Stop Sequence = {stop_sequence}]")
        output_lines.append(f"Result:\n{content}\n")
    except Exception as e:
        output_lines.append(f"[Stop Sequence] Error: {e}\n")

    # Save outputs
    output_text = "\n".join(output_lines)
    with open("outputs/parameter_experiments.txt", "w", encoding="utf-8") as f:
        f.write(output_text)
    logging.info("Parameter experiments successfully written to outputs/parameter_experiments.txt")
    print(output_text)

if __name__ == "__main__":
    run_experiments()
