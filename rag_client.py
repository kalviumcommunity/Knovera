import os
import logging
import json
from dotenv import load_dotenv
from openai import OpenAI, AuthenticationError, RateLimitError, APIError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api_exchange.log"),
        logging.StreamHandler()
    ]
)

def main():
    # Task 1: Load config from .env
    load_dotenv()
    
    api_key = os.getenv("OPENROUTER_API_KEY", os.getenv("OPENAI_API_KEY"))
    base_url = os.getenv("OPENROUTER_BASE_URL", os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"))
    model_name = os.getenv("OPENROUTER_MODEL", os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"))
    
    if not api_key:
        logging.error("Missing OPENROUTER_API_KEY or OPENAI_API_KEY in .env file")
        return
        
    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )
    
    messages = [
        {"role": "system", "content": "You are a helpful RAG assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
    
    # Task 3: Log the outgoing request
    logging.info(f"Sending request to model: {model_name}")
    logging.info(f"Outgoing messages payload: {json.dumps(messages, indent=2)}")
    
    try:
        # Task 2: Send request
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.7
        )
        
        reply_content = response.choices[0].message.content
        
        # Task 3: Log the incoming response and token usage
        logging.info(f"Incoming response payload: {response.model_dump_json(indent=2)}")
        logging.info(f"Token usage: {response.usage}")
        
        # Task 2: Print the response text
        print("\n--- Assistant Reply ---")
        print(reply_content)
        print("-----------------------\n")
        
    # Task 4: Handle errors clearly
    except AuthenticationError:
        logging.error("Authentication Failed (401): Please check your API key. It may be invalid or expired.")
    except RateLimitError:
        logging.error("Rate Limit Exceeded (429): You are sending requests too quickly or have exhausted your quota.")
    except APIError as e:
        logging.error(f"API Error: An error occurred while contacting the API - {e}")
    except Exception as e:
        logging.error(f"Unexpected Error: {e}")

if __name__ == "__main__":
    main()
