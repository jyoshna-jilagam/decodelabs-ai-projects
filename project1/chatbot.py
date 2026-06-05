"""
Rule-Based AI Chatbot — Project 1
DecodeLabs Industrial Training | Batch 2026
Architecture: IPO Model with O(1) Dictionary Lookup
"""

# ── Knowledge Base ─────────────────────────────────────────────────────────────
RESPONSES = {
    # Greetings
    "hello"         : "Hello! I'm DecodeLabs Assistant. How can I help you today?",
    "hi"            : "Hey there! What would you like to know?",
    "hey"           : "Hey! Ask me anything.",

    # Identity
    "who are you"   : "I'm a Rule-Based AI Chatbot built at DecodeLabs — powered by pure logic, zero hallucinations.",
    "what are you"  : "I'm an AI chatbot that uses deterministic if-dict logic, not machine learning.",
    "who made you"  : "I was engineered by an intern at DecodeLabs as Project 1.",

    # AI Concepts
    "what is ai"    : "AI is the simulation of human intelligence by machines using logic, learning, or reasoning.",
    "what is a rule based chatbot" : "A rule-based chatbot responds using hard-coded if-else or dictionary logic — 100% predictable, zero hallucination risk.",
    "what is machine learning" : "Machine learning lets systems learn patterns from data instead of following explicit rules.",
    "what is a llm" : "A Large Language Model (LLM) is a probabilistic AI trained on massive text data to generate human-like responses.",
    "what is nlp"   : "NLP (Natural Language Processing) is AI's ability to understand and generate human language.",

    # Small talk
    "how are you"   : "Running at full capacity! All logic gates are green.",
    "what can you do" : "I can answer questions about AI, chat with you, and demonstrate rule-based logic. Type 'help' to see topics.",
    "help"          : "Try asking: 'what is AI', 'who are you', 'what is machine learning', 'what is a LLM', or just say hello!",
    "tell me a joke" : "Why do programmers prefer dark mode? Because light attracts bugs!",
    "thank you"     : "You're welcome! Anything else?",
    "thanks"        : "Happy to help!",

    # Farewell
    "bye"           : "Goodbye! Keep building.",
    "goodbye"       : "See you next time. Stay curious!",
    "quit"          : "Exiting... Goodbye!",
}

EXIT_COMMANDS = {"exit", "quit", "bye", "goodbye"}

# ── Core Functions ──────────────────────────────────────────────────────────────

def sanitize(raw: str) -> str:
    """Normalize input: strip whitespace and convert to lowercase."""
    return raw.lower().strip()


def get_response(user_input: str) -> str:
    """
    Look up a response from the knowledge base in O(1) time.
    Falls back to a default message if the intent is unknown.
    """
    return RESPONSES.get(user_input, "I don't understand that yet. Try typing 'help' to see what I can do.")


def is_exit(user_input: str) -> bool:
    """Check if the user wants to end the session."""
    return user_input in EXIT_COMMANDS


# ── Main Loop ──────────────────────────────────────────────────────────────────

def chatbot():
    """
    Main chatbot loop.
    IPO Model:
      INPUT   → raw user text
      PROCESS → sanitize → exit check → O(1) dict lookup
      OUTPUT  → response printed to console
    """
    print("=" * 50)
    print("  DecodeLabs AI Assistant  |  Project 1")
    print("  Type 'help' for topics  |  'exit' to quit")
    print("=" * 50)

    while True:
        raw_input_text = input("\nYou: ")
        clean_input = sanitize(raw_input_text)

        if not clean_input:
            continue

        if is_exit(clean_input):
            print(f"Bot: {RESPONSES.get(clean_input, 'Goodbye!')}")
            break

        response = get_response(clean_input)
        print(f"Bot: {response}")


if __name__ == "__main__":
    chatbot()
