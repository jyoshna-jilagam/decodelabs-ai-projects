# Project 1 — Rule-Based AI Chatbot 🤖

> **DecodeLabs Industrial Training | Batch 2026**

---

## 🎯 Goal

Build a rule-based chatbot that responds to predefined user inputs using pure control flow and logic — no machine learning required.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────┐
│              while True Loop                │
│                                             │
│  INPUT ──► Sanitize ──► Exit? ──► Dict      │
│   raw       .lower()    break    .get()     │
│   text      .strip()             O(1)       │
│                                    │        │
│                               RESPONSE      │
└─────────────────────────────────────────────┘
```

**IPO Model:**
- **Input** — Raw user text captured via `input()`
- **Process** — Sanitize → Check exit → O(1) dictionary lookup
- **Output** — Matched response or fallback message

---

## 🔑 Key Design Decisions

### Why Dictionary over if-elif?

| Approach | Complexity | Maintenance |
|----------|-----------|-------------|
| if-elif ladder | O(n) — scans every condition | High technical debt |
| Dictionary `.get()` | **O(1)** — direct hash lookup | Clean, scalable |

The `.get(key, default)` method handles both **lookup and fallback in a single atomic operation**.

### Sanitization
```python
clean_input = raw_input.lower().strip()
```
Normalizes `"Hello"`, `"HELLO"`, `"  hello  "` → all map to the same key.

---

## 📋 Specification Checklist

- [x] Continuous `while True` input loop
- [x] Input sanitization (case + whitespace)
- [x] Knowledge base with 15+ intents
- [x] Fallback response for unknowns
- [x] Clean `break` exit on "exit"/"bye"/"quit"/"goodbye"
- [x] Empty input guard (`if not clean_input: continue`)
- [x] Modular functions with docstrings

---

## ▶️ How to Run

```bash
python project1/chatbot.py
```

### Sample Session

```
==================================================
  DecodeLabs AI Assistant  |  Project 1
  Type 'help' for topics  |  'exit' to quit
==================================================

You: hello
Bot: Hello! I'm DecodeLabs Assistant. How can I help you today?

You: what is ai
Bot: AI is the simulation of human intelligence by machines using logic, learning, or reasoning.

You: tell me a joke
Bot: Why do programmers prefer dark mode? Because light attracts bugs!

You: bye
Bot: Goodbye! Keep building.
```

---

## 📂 Files

| File | Description |
|------|-------------|
| `chatbot.py` | Main chatbot script |
| `README.md` | This documentation |

---

## 🧠 Concepts Covered

- Control flow & decision-making logic
- Hash maps / Python dictionaries
- Input sanitization & normalization
- IPO (Input-Process-Output) model
- White-box AI (deterministic, traceable, zero hallucination)
