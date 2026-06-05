# System Architecture Notes

> DecodeLabs AI Projects — Batch 2026

---

## Project 1: Rule-Based AI Chatbot

### Design Pattern: White-Box IPO

```
Input → Sanitization → Process → Output
         .lower()      dict        print
         .strip()      .get()
```

### Key Insight: O(1) vs O(n)

The if-elif ladder checks every condition linearly.
A Python dictionary uses a hash map internally — lookup is O(1) regardless of catalogue size.

```python
# Anti-pattern — O(n) if-elif ladder
if user_input == "hello":    response = "Hi!"
elif user_input == "bye":    response = "Goodbye!"
elif user_input == "help":   response = "..."
# ... scales poorly

# Professional pattern — O(1) hash lookup
response = RESPONSES.get(user_input, "I don't understand.")
```

### Exit Strategy
Multiple exit keywords all map to EXIT_COMMANDS set for O(1) membership check:
```python
EXIT_COMMANDS = {"exit", "quit", "bye", "goodbye"}
if clean_input in EXIT_COMMANDS: break
```

---

## Project 3: AI Recommendation Engine

### Design Pattern: Content-Based Filtering Pipeline

```
User Skills → TF-IDF Vectors → Cosine Score → Sort → Top-N
```

### TF-IDF: Why It Beats Binary Matching

Binary (Jaccard): treats "python" and "kubernetes" equally if both present.
TF-IDF: "kubernetes" gets higher weight because it appears in fewer job roles (higher IDF).

```
IDF("python")     = log(12/10) = 0.18   ← common, penalized
IDF("kubernetes") = log(12/2)  = 1.79   ← rare, rewarded
```

### Cosine Similarity: Why Not Euclidean?

Euclidean distance is magnitude-sensitive.
A user with 3 skills vs 10 skills would score differently even with identical interests.
Cosine measures angle (direction of interest) not magnitude (quantity of skills).

```
cos(θ) = (A · B) / (||A|| × ||B||)
       = dot product / (magnitude A × magnitude B)
```

### Cold-Start Guard

```python
if mag_a == 0 or mag_b == 0:
    return 0.0   # zero vector → no match possible
```

Prevents ZeroDivisionError when a new user has no skills or a new item has no tags.

---

## Shared Principles

| Principle | P1 Implementation | P3 Implementation |
|-----------|------------------|------------------|
| IPO Model | input→sanitize→dict→print | skills→vectorize→score→top-n |
| No external libs | stdlib only | math module only |
| Fallback handling | .get() default | cold-start guard |
| Modular design | 3 pure functions | 5 pure functions |
| Docstrings | All functions | All functions |
