# Project 3 — AI Recommendation Logic 🎯

> **DecodeLabs Industrial Training | Batch 2026**

---

## 🎯 Goal

Build a content-based filtering recommendation engine that maps user skills to career paths using TF-IDF vectorization and Cosine Similarity — no external ML libraries required.

---

## 🏗️ Architecture

```
User Skills Input
      │
      ▼
┌─────────────────┐
│  Step 1:        │  .lower().strip() each skill
│  INGESTION      │  Build TF-IDF user profile vector
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Step 2:        │  Loop through all 12 job roles
│  SCORING        │  cosine_similarity(user_vec, item_vec)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Step 3:        │  sorted(..., reverse=True)
│  SORTING        │  Highest score → top of list
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Step 4:        │  ranked[:top_n]
│  FILTERING      │  Return Top-3 only (prevents choice overload)
└─────────────────┘
         │
         ▼
  Ranked Recommendations
```

---

## 📐 Math Behind It

### TF-IDF Weighting

```
TF(t)  = count(t in document) / total terms in document
IDF(t) = log(Total documents / documents containing t)
Weight = TF × IDF
```

> Specific skills (e.g. "kubernetes") score higher than generic ones (e.g. "python") because IDF penalizes terms appearing in many documents.

### Cosine Similarity

```
cos(θ) = (A · B) / (||A|| × ||B||)
```

| Score | Meaning |
|-------|---------|
| 1.0 | Perfect match — identical orientation |
| 0.5 | Moderate match |
| 0.0 | No common features (orthogonal vectors) |

> Magnitude-invariant: a user with 3 skills vs 10 skills is judged on **direction**, not size.

---

## 📋 Specification Checklist

- [x] Accepts minimum 3 user skill inputs (data density requirement)
- [x] TF-IDF feature extraction (no sklearn — pure math)
- [x] Cosine Similarity scoring for all 12 job roles
- [x] Sorted descending by score
- [x] Top-3 filtered output (choice overload prevention)
- [x] Cold-start guard (`mag == 0` check)
- [x] Onboarding survey bypass for cold-start problem
- [x] Visual score bar in output
- [x] Modular functions with docstrings

---

## ▶️ How to Run

```bash
python project3/recommender.py
```

### Sample Session

```
====================================================
  DecodeLabs Tech Stack Recommender | P3
  Content-Based Filtering | TF-IDF + Cosine
====================================================

Enter your skills one by one (min 3). Type 'done' when finished.
  Skill 1: Python
  Skill 2: Machine Learning
  Skill 3: TensorFlow
  Skill 4: done

====================================================
  TOP CAREER RECOMMENDATIONS FOR YOUR PROFILE
====================================================
  Skills: Python, Machine Learning, TensorFlow
----------------------------------------------------
  1. Machine Learning Engineer
     Match: 78.43%  ████████████████
  2. AI Engineer
     Match: 71.20%  ██████████████
  3. Data Scientist
     Match: 52.18%  ██████████
====================================================
```

---

## 🆚 Why Content-Based over Collaborative Filtering?

| Feature | Collaborative | Content-Based (Ours) |
|---------|--------------|----------------------|
| Data needed | Historical user interactions | Item attributes only |
| Cold start | ❌ Fails without user history | ✅ Works immediately |
| Transparency | Black box | White box — explainable |
| Scale needed | Large dataset | Works with small catalogue |

---

## 📂 Files

| File | Description |
|------|-------------|
| `recommender.py` | Main recommendation engine |
| `README.md` | This documentation |

---

## 🧠 Concepts Covered

- Content-based filtering
- TF-IDF vectorization (from scratch)
- Cosine similarity (from scratch)
- Vector space model
- Cold-start problem & bypass strategies
- Top-N ranking pipeline
