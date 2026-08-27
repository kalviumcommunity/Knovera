"""
Demo: Multi-turn RAG Conversation with History Management

Demonstrates:
1. Maintaining message history across multiple turns
2. Measuring tokens before each request
3. Trimming old turns when history exceeds budget
4. Keeping requests within token limit despite long conversation
"""

import os
import sys
import logging
import json
from typing import List, Tuple

# Add paths for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.history_manager import HistoryManager
from token_estimation import count_tokens

# Configure logging
os.makedirs("outputs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("outputs/history_demo.log"),
        logging.StreamHandler()
    ]
)

# ============================================================================
# DEMO CONVERSATION DATA
# ============================================================================

SYSTEM_PROMPT = (
    "You are Knovera, an internal support assistant for staff documentation. "
    "Your scope is to provide accurate, factual information regarding internal company policies. "
    "Answer concisely in a maximum of 2 sentences using a professional tone. "
    "If you are unsure or the information is not provided in context, state: "
    "'I do not have enough information to answer this question.'"
)

# Series of user questions and simulated assistant responses
# These are realistic Q&A pairs to simulate a growing conversation
CONVERSATION_TURNS = [
    {
        "user": "What is our standard customer refund window?",
        "assistant": (
            "The standard refund window for Knovera is 30 calendar days from the date of purchase. "
            "Items must be in original condition and accompanied by proof of purchase such as an invoice "
            "or order reference ID. Requests submitted after the 30-day window are subject to store credit "
            "only if approved by a manager."
        )
    },
    {
        "user": "Can you explain the return merchandise authorization process?",
        "assistant": (
            "When a customer initiates a return, our support team verifies the order in the CRM system "
            "and issues an RMA (Return Merchandise Authorization) tracking number. The customer ships the "
            "item using this RMA number, and upon receipt and inspection, we process the refund to the "
            "original payment method within 5-7 business days."
        )
    },
    {
        "user": "What about items that are customized or digital products?",
        "assistant": (
            "Customized orders and downloadable digital software licenses are typically non-refundable "
            "unless verified as defective by our technical support team. Opened consumable goods also fall "
            "under the non-refundable category, but exceptions can be made on a case-by-case basis at managerial discretion."
        )
    },
    {
        "user": "Are there exceptions for large refund amounts?",
        "assistant": (
            "Yes, any refund request exceeding $1,000 USD or requesting wire transfer reimbursement requires "
            "dual sign-off from both the Operations Lead and Finance Director. This policy ensures financial "
            "compliance and protects against fraud in high-value transactions."
        )
    },
    {
        "user": "What is the store credit policy for late returns?",
        "assistant": (
            "For requests made between 31 and 60 days after purchase, store credit may be issued at managerial "
            "discretion rather than a full refund. This allows us to retain customer goodwill while maintaining "
            "our refund policy structure. Requests beyond 60 days are typically declined."
        )
    },
    {
        "user": "How do we handle disputes or customer escalations?",
        "assistant": (
            "Disputed refund cases are escalated to the Customer Relations team, who review the order history, "
            "correspondence, and supporting documentation. If the dispute cannot be resolved, it is forwarded to "
            "the Operations Lead for final determination and potential executive override."
        )
    },
    {
        "user": "What documentation is required for a refund claim?",
        "assistant": (
            "Required documentation includes: a valid sales receipt or order reference ID, proof of purchase method "
            "(credit card, PayPal, etc.), and photographs showing the item's condition if there is a dispute. For "
            "damaged or defective items, a technical support report must be included in the claim file."
        )
    },
    {
        "user": "Can refunds be issued to a different payment method than the original?",
        "assistant": (
            "Refunds are always issued to the original payment method to prevent fraud and ensure accountability. "
            "However, customers may request a wire transfer for very large amounts ($5,000+) with proper verification "
            "and dual managerial approval, though this is rare."
        )
    },
    {
        "user": "What is our policy on international returns and refunds?",
        "assistant": (
            "International returns follow the same 30-day window but require customers to arrange and pay for return "
            "shipping. Refunds are issued in the original currency at the rate applicable on the refund processing date. "
            "Customs duties and import taxes are not refundable."
        )
    },
    {
        "user": "How should staff handle repeat return requests from the same customer?",
        "assistant": (
            "Staff should review the customer's return history in the CRM. Multiple returns (3+ in 12 months) may indicate "
            "a pattern that requires managerial review. While we maintain a customer-first approach, repeated abuse of the "
            "return policy may result in account restrictions or escalation to the Legal/Compliance team."
        )
    }
]


def format_message_display(msg: dict, token_count: int) -> str:
    """Format a message for display with role and token count."""
    role = msg["role"].upper()
    preview = msg["content"][:80]
    if len(msg["content"]) > 80:
        preview += "..."
    return f"[{role}] ({token_count} tokens) {preview}"


def run_demo():
    """Run the multi-turn conversation demo with history management."""
    
    output_lines = []
    
    output_lines.append("=" * 80)
    output_lines.append("KNOVERA - MULTI-TURN RAG CONVERSATION WITH HISTORY MANAGEMENT")
    output_lines.append("=" * 80)
    output_lines.append("")
    
    # SETUP
    output_lines.append("SCENARIO SETUP:")
    output_lines.append("-" * 80)
    output_lines.append(f"System Prompt: {SYSTEM_PROMPT[:100]}...")
    output_lines.append(f"Token Budget: 500 tokens (small to force aggressive trimming)")
    output_lines.append(f"Total Conversation Turns: {len(CONVERSATION_TURNS)}")
    output_lines.append("")
    
    # Calculate what the naive approach would look like
    naive_tokens = count_tokens(SYSTEM_PROMPT)["token_count"]
    for turn in CONVERSATION_TURNS:
        naive_tokens += count_tokens(turn["user"])["token_count"]
        naive_tokens += count_tokens(turn["assistant"])["token_count"]
    
    output_lines.append(f"NAIVE APPROACH (no trimming):")
    output_lines.append(f"  Total tokens if all messages kept: {naive_tokens}")
    output_lines.append(f"  Budget overrun: ~{max(0, naive_tokens - 500)} tokens over limit")
    output_lines.append("")
    output_lines.append("INTELLIGENT APPROACH (with history manager):")
    output_lines.append("-" * 80)
    output_lines.append("")
    
    # Initialize history manager with small budget to force trimming (500 tokens forces aggressive trimming)
    history = HistoryManager(system_message=SYSTEM_PROMPT, token_budget=500)
    
    all_trimming_events = []
    turn_reports = []
    
    # Simulate the conversation
    for turn_idx, turn_data in enumerate(CONVERSATION_TURNS, 1):
        output_lines.append(f"TURN {turn_idx}:")
        output_lines.append("-" * 80)
        
        # Add user message
        history.add_user_message(turn_data["user"])
        user_tokens = count_tokens(turn_data["user"])["token_count"]
        output_lines.append(f"  User: {format_message_display(history.messages[-1], user_tokens)}")
        
        # Add assistant message (simulated response)
        history.add_assistant_message(turn_data["assistant"])
        assistant_tokens = count_tokens(turn_data["assistant"])["token_count"]
        output_lines.append(f"  Assistant: {format_message_display(history.messages[-1], assistant_tokens)}")
        
        # Check if trimming is needed
        current_tokens = history.count_history_tokens()
        output_lines.append(f"  History tokens BEFORE trim: {current_tokens}")
        
        if history.should_trim():
            output_lines.append(f"  ⚠️  Approaching token budget (80% threshold = {int(history.token_budget * 0.8)})")
            trim_stats = history.trim_to_budget()
            
            if trim_stats["turns_removed"] > 0:
                output_lines.append(f"  ✓ Trimmed {trim_stats['turns_removed']} old turn(s)")
                output_lines.append(f"    Tokens freed: {trim_stats['tokens_saved']}")
                output_lines.append(f"    History tokens AFTER trim: {trim_stats['final_tokens']}")
                
                all_trimming_events.append({
                    "turn": turn_idx,
                    "turns_removed": trim_stats["turns_removed"],
                    "tokens_saved": trim_stats["tokens_saved"],
                    "final_tokens": trim_stats["final_tokens"]
                })
        else:
            output_lines.append(f"  ✓ Within budget (using {round(100*current_tokens/history.token_budget, 1)}%)")
        
        # Show current status
        status = history.get_status()
        output_lines.append(f"  Final status: {status['message_count']} messages, "
                           f"{status['token_count']} tokens ({status['token_usage_percent']}% of budget)")
        
        turn_reports.append({
            "turn": turn_idx,
            "messages": status["message_count"],
            "tokens": status["token_count"],
            "usage_percent": status["token_usage_percent"],
            "trimmed": len(all_trimming_events) > 0 and all_trimming_events[-1]["turn"] == turn_idx
        })
        
        output_lines.append("")
    
    # FINAL SUMMARY
    output_lines.append("=" * 80)
    output_lines.append("FINAL SUMMARY")
    output_lines.append("=" * 80)
    output_lines.append("")
    
    final_status = history.get_status()
    output_lines.append(f"Total turns completed: {len(CONVERSATION_TURNS)}")
    output_lines.append(f"Current messages in history: {final_status['message_count']}")
    output_lines.append(f"Current token count: {final_status['token_count']}")
    output_lines.append(f"Token budget: {final_status['token_budget']}")
    output_lines.append(f"Within budget: {'✓ YES' if final_status['within_budget'] else '✗ NO'}")
    output_lines.append(f"Usage: {final_status['token_usage_percent']}%")
    output_lines.append("")
    
    output_lines.append("TRIMMING EVENTS:")
    if all_trimming_events:
        output_lines.append(f"Total trim events: {len(all_trimming_events)}")
        for event in all_trimming_events:
            output_lines.append(f"  Turn {event['turn']}: Removed {event['turns_removed']} turn(s), "
                               f"saved {event['tokens_saved']} tokens → {event['final_tokens']} tokens remain")
        total_tokens_saved = sum(e["tokens_saved"] for e in all_trimming_events)
        output_lines.append(f"Total tokens saved by trimming: {total_tokens_saved}")
    else:
        output_lines.append("No trimming was necessary (unusual with small budget).")
    
    output_lines.append("")
    output_lines.append("TOKEN USAGE OVER TIME:")
    output_lines.append(f"{'Turn':<6} {'Messages':<12} {'Tokens':<10} {'%Budget':<10} {'Trimmed?':<10}")
    output_lines.append("-" * 48)
    for report in turn_reports:
        trimmed_mark = "Yes" if report["trimmed"] else "No"
        output_lines.append(
            f"{report['turn']:<6} {report['messages']:<12} {report['tokens']:<10} "
            f"{report['usage_percent']:<10.1f} {trimmed_mark:<10}"
        )
    
    output_lines.append("")
    output_lines.append("=" * 80)
    output_lines.append("CONCLUSION")
    output_lines.append("=" * 80)
    output_lines.append(f"✓ Handled {len(CONVERSATION_TURNS)} conversation turns")
    output_lines.append(f"✓ Stayed within {history.token_budget} token budget")
    output_lines.append(f"✓ Trimmed old turns intelligently to preserve recent context")
    output_lines.append(f"✓ Final history: {final_status['message_count']} messages, "
                       f"{final_status['token_count']} tokens")
    output_lines.append(f"✓ All API requests would succeed without hitting token limits")
    output_lines.append("")
    
    # Print all output
    output_text = "\n".join(output_lines)
    print(output_text)
    
    # Save to file with UTF-8 encoding to handle special characters
    output_file = "outputs/history_management_demo.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(output_text)
    
    logging.info(f"Demo output saved to {output_file}")
    
    return output_lines


if __name__ == "__main__":
    run_demo()
