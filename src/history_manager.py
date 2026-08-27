"""
History Manager for Multi-turn RAG Conversations

Maintains conversation history, tracks token usage, and implements trimming/summarization
strategies to keep requests within token budget.
"""

import logging
from typing import List, Dict, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from token_estimation import count_tokens

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class HistoryManager:
    """
    Manages multi-turn conversation history with token tracking and trimming.
    
    Responsibilities:
    1. Maintain message history (system + alternating user/assistant messages)
    2. Measure tokens before each request
    3. Implement trim/summarization when approaching token budget
    4. Preserve system message and recent context
    """
    
    def __init__(self, system_message: str, token_budget: int = 2000):
        """
        Initialize history manager.
        
        Args:
            system_message: The system prompt that guides assistant behavior
            token_budget: Maximum tokens allowed for history + request (default: 2000)
        """
        self.system_message = system_message
        self.token_budget = token_budget
        self.messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_message}
        ]
        self.trim_summary = None  # Store summary of trimmed messages
        self.turn_count = 0
        
        # Track metrics for reporting
        self.system_tokens = count_tokens(system_message)["token_count"]
        logging.info(f"HistoryManager initialized with token budget: {token_budget}, "
                    f"system message tokens: {self.system_tokens}")
    
    def add_user_message(self, content: str) -> None:
        """Add a user message to history."""
        self.messages.append({"role": "user", "content": content})
        logging.debug(f"Added user message (turn {self.turn_count + 1}): {len(content)} chars")
    
    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message to history."""
        self.messages.append({"role": "assistant", "content": content})
        self.turn_count += 1
        logging.debug(f"Added assistant message (turn {self.turn_count}): {len(content)} chars")
    
    def count_history_tokens(self) -> int:
        """
        Count total tokens in current message history.
        
        Returns:
            Total token count for all messages
        """
        total_tokens = 0
        for msg in self.messages:
            tokens = count_tokens(msg["content"])["token_count"]
            total_tokens += tokens
        return total_tokens
    
    def get_history_breakdown(self) -> Dict:
        """
        Get detailed breakdown of tokens per message.
        
        Returns:
            Dictionary with token counts for each message
        """
        breakdown = {}
        for i, msg in enumerate(self.messages):
            role = msg["role"]
            tokens = count_tokens(msg["content"])["token_count"]
            breakdown[f"{i:02d}_{role}"] = tokens
        return breakdown
    
    def should_trim(self) -> bool:
        """
        Check if history is approaching or exceeding token budget.
        
        Returns:
            True if tokens exceed 80% of budget
        """
        current_tokens = self.count_history_tokens()
        threshold = int(self.token_budget * 0.8)
        return current_tokens > threshold
    
    def trim_oldest_turn(self) -> Optional[tuple]:
        """
        Remove the oldest user-assistant pair (preserving system message and most recent exchange).
        
        Returns:
            Tuple of (removed_user_msg, removed_assistant_msg) or None if nothing to remove
        """
        # Keep at least system message + current user-assistant pair
        if len(self.messages) <= 3:
            logging.warning("Cannot trim: only system + current turn remain")
            return None
        
        # Remove the second and third messages (oldest user-assistant pair)
        removed_user = self.messages.pop(1)
        removed_assistant = self.messages.pop(1)  # Index shifts after first pop
        
        logging.info(f"Trimmed oldest turn: user ({len(removed_user['content'])} chars) "
                    f"+ assistant ({len(removed_assistant['content'])} chars)")
        
        return (removed_user, removed_assistant)
    
    def trim_to_budget(self, max_iterations: int = 10) -> Dict:
        """
        Trim history until it fits within budget.
        
        Args:
            max_iterations: Prevent infinite loops
            
        Returns:
            Dictionary with trimming stats
        """
        stats = {
            "initial_tokens": self.count_history_tokens(),
            "turns_removed": 0,
            "messages_removed": []
        }
        
        iterations = 0
        while self.should_trim() and iterations < max_iterations:
            trimmed = self.trim_oldest_turn()
            if trimmed is None:
                break
            
            removed_user, removed_assistant = trimmed
            stats["messages_removed"].append({
                "user": removed_user["content"][:100],  # Truncate for logging
                "assistant": removed_assistant["content"][:100]
            })
            stats["turns_removed"] += 1
            iterations += 1
        
        stats["final_tokens"] = self.count_history_tokens()
        stats["tokens_saved"] = stats["initial_tokens"] - stats["final_tokens"]
        
        if stats["turns_removed"] > 0:
            logging.info(f"Trimmed {stats['turns_removed']} turns, "
                        f"freed {stats['tokens_saved']} tokens "
                        f"({stats['initial_tokens']} → {stats['final_tokens']})")
        
        return stats
    
    def get_messages(self) -> List[Dict[str, str]]:
        """Return the current message history for API call."""
        return self.messages.copy()
    
    def get_status(self) -> Dict:
        """
        Get comprehensive status report of current history.
        
        Returns:
            Dictionary with status metrics
        """
        current_tokens = self.count_history_tokens()
        
        return {
            "message_count": len(self.messages),
            "turn_count": self.turn_count,
            "token_count": current_tokens,
            "token_budget": self.token_budget,
            "token_usage_percent": round(100 * current_tokens / self.token_budget, 1),
            "within_budget": current_tokens <= self.token_budget,
            "system_tokens": self.system_tokens,
            "messages": [
                {
                    "role": msg["role"],
                    "content_preview": msg["content"][:50] + ("..." if len(msg["content"]) > 50 else ""),
                    "tokens": count_tokens(msg["content"])["token_count"]
                }
                for msg in self.messages
            ]
        }
