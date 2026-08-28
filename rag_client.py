import os
import logging
import json
from dotenv import load_dotenv
from openai import OpenAI, AuthenticationError, RateLimitError, APIError
import sys

from src.history_manager import HistoryManager
from structured_output import parse
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
    
    # Initialize HistoryManager
    # Initialize HistoryManager with strict JSON instructions
    system_message = (
        'You are a helpful RAG assistant. '
        'Reply with ONLY a JSON object: '
        '{"answer": string, "source": string}. No extra text or prose.'
    )
    history = HistoryManager(system_message=system_message, token_budget=4000)
    
    print("Welcome to Knovera RAG Assistant!")
    print("Type 'exit' or 'quit' to end the conversation.\n")
    
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ['exit', 'quit']:
                break
            if not user_input.strip():
                continue
                
            history.add_user_message(user_input)
            
            # Trim if needed before making the request
            if history.should_trim():
                history.trim_to_budget()
                
            messages = history.get_messages()
            
            # Task 3: Log the outgoing request
            logging.info(f"Sending request to model: {model_name}")
            logging.info(f"Outgoing messages payload: {json.dumps(messages, indent=2)}")
            
            # Task 2: Send request
            # Task 2: Send request with JSON mode enforcement
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=300,
                    response_format={"type": "json_object"},
                    stop=["\n\nUser:", "User:"]
                )
            except Exception as e:
                logging.warning(f"JSON mode failed ({e}), falling back to standard request")
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=300,
                    stop=["\n\nUser:", "User:"]
                )
            
            reply_content = response.choices[0].message.content
            
            # Add assistant message to history
            history.add_assistant_message(reply_content)
            
            # Task 3: Log the incoming response and token usage
            logging.info(f"Incoming response payload: {response.model_dump_json(indent=2)}")
            logging.info(f"Token usage: {response.usage}")
            
            # Parse and validate the response using the logic from structured_output.py
            parsed_data, err = parse(reply_content)
            
            if parsed_data:
                print(f"\nAssistant (JSON): {json.dumps(parsed_data, indent=2)}\n")
            else:
                print(f"\nAssistant (Raw, parsing failed - {err}): {reply_content}\n")
            
        except AuthenticationError:
            logging.error("Authentication Failed (401): Please check your API key. It may be invalid or expired.")
            break
        except RateLimitError:
            logging.error("Rate Limit Exceeded (429): You are sending requests too quickly or have exhausted your quota.")
            break
        except APIError as e:
            logging.error(f"API Error: An error occurred while contacting the API - {e}")
            break
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            logging.error(f"Unexpected Error: {e}")
            break

if __name__ == "__main__":
    main()
