# Prompt Construction & Role Analysis (Assignment 3.13)

## 1. System vs. User Roles
In OpenAI Chat Completion models, messages are divided into distinct roles:
- **`system`**: Sets the overarching persona, operational scope, tone, behavioral rules, length limits, and fallback constraints. The system message acts as the control panel that guides the assistant's behavior across turns.
- **`user`**: Represents the current query, task, or input provided for a specific interaction.

## 2. System Message Design
Our chosen system prompt in `prompts/system_prompt.txt` is:
> *"You are Knovera, an internal support assistant for staff documentation. Your scope is to provide accurate, factual information regarding internal company policies. Answer concisely in a maximum of 2 sentences using a professional tone. If you are unsure or the information is not provided in context, state: 'I do not have enough information to answer this question.'"*

### Key Components:
- **Role**: Support Assistant for Knovera internal documentation.
- **Scope**: Internal company policies and factual guidance only.
- **Tone & Length Constraints**: Professional, max 2 sentences.
- **Fallback / Refusal Rule**: Explicit refusal phrasing ("I do not have enough information to answer this question") when context or information is missing.

---

## 3. Comparison of Prompt Variations

| Variation | User Prompt | Characteristics | Observed Output / Behavior |
| :--- | :--- | :--- | :--- |
| **Variation 1 (Vague)** | `"Explain our refund policy."` | Ambiguous, no format or length constraints. | Model defaults to generic, potentially rambling explanations or hallucinated default policies. |
| **Variation 2 (Clear & Constrained)** | `"In one concise sentence, state the standard refund window in days and key eligibility criteria."` | Specifies exact task, unit (days), focus criteria, and single-sentence length constraint. | Produces a tight, direct, factual answer that strictly adheres to the requested parameters. |
| **Variation 3 (Explicit Format)** | `"State our refund window. Reply ONLY with a valid JSON object matching this schema: {\"refund_window_days\": int, \"policy_summary\": string}."` | Enforces structural output format suitable for automated API parsing. | Outputs clean JSON without conversational filler text. |

---

## 4. Why the Chosen Prompt Works (Task 4 Documentation)
Prompt Variation 2 (and Variation 3 for structured data APIs) is far superior to Prompt Variation 1 for the following reasons:
1. **Eliminates Ambiguity**: Explicitly names the target metric (`refund window in days`) rather than asking open-ended questions (`explain our policy`).
2. **Strict Output Formatting**: Enforces length constraints (`one concise sentence` / `valid JSON`), preventing model verbose rambling.
3. **Reduces Hallucination**: Clear boundaries paired with the system message refusal rule force the model to stay grounded and state when details are missing rather than guessing.
4. **Machine Parsability**: Constraining the response structure guarantees predictable down-stream integration in the application pipeline.
