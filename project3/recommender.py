"""
AI Recommendation Logic — Project 3
DecodeLabs Industrial Training | Batch 2026
Architecture: Content-Based Filtering | TF-IDF Vectorization + Cosine Similarity
Pipeline: Ingestion → Scoring → Sorting → Filtering (Top-N)
"""

import math

# ── Item Catalogue (Knowledge Base) ───────────────────────────────────────────
CATALOGUE = {
    "Data Scientist"            : ["python", "sql", "machine learning", "statistics", "data analysis", "pandas", "numpy", "visualization"],
    "Machine Learning Engineer" : ["python", "machine learning", "deep learning", "tensorflow", "pytorch", "algorithms", "mlops"],
    "Data Analyst"              : ["sql", "excel", "python", "data analysis", "visualization", "statistics", "reporting"],
    "Backend Developer"         : ["python", "java", "sql", "apis", "databases", "docker", "rest", "node"],
    "Frontend Developer"        : ["javascript", "react", "html", "css", "typescript", "ui", "ux", "figma"],
    "Full Stack Developer"      : ["javascript", "python", "react", "node", "sql", "apis", "html", "css", "docker"],
    "DevOps Engineer"           : ["docker", "kubernetes", "aws", "ci/cd", "linux", "terraform", "ansible", "cloud"],
    "Cloud Architect"           : ["aws", "azure", "cloud", "kubernetes", "terraform", "networking", "security", "microservices"],
    "Cybersecurity Analyst"     : ["networking", "linux", "security", "ethical hacking", "firewalls", "python", "cryptography"],
    "AI Engineer"               : ["python", "deep learning", "nlp", "machine learning", "tensorflow", "transformers", "llm", "apis"],
    "Mobile Developer"          : ["flutter", "react", "kotlin", "swift", "java", "apis", "firebase", "ui"],
    "Data Engineer"             : ["python", "sql", "spark", "hadoop", "etl", "kafka", "cloud", "databases", "airflow"],
}

# ── Phase 1: TF-IDF Vectorizer ────────────────────────────────────────────────

def compute_idf(catalogue: dict) -> dict:
    """
    Compute IDF for every unique skill across the full catalogue.
    IDF(t) = log(Total_docs / docs_containing_t)
    Rare, specific skills get higher weight; generic skills are penalized.
    """
    N = len(catalogue)
    doc_freq = {}
    for tags in catalogue.values():
        for skill in set(tags):
            doc_freq[skill] = doc_freq.get(skill, 0) + 1
    return {skill: math.log(N / freq) for skill, freq in doc_freq.items()}


def vectorize(tags: list, idf: dict) -> dict:
    """
    Convert a skill list into a TF-IDF weighted sparse vector.
    TF(t) = count(t in doc) / total_terms_in_doc
    Vector = {skill: TF * IDF}
    """
    total = len(tags)
    tf = {}
    for skill in tags:
        tf[skill] = tf.get(skill, 0) + 1
    return {skill: (count / total) * idf.get(skill, 0)
            for skill, count in tf.items()}


# ── Phase 2: Cosine Similarity ────────────────────────────────────────────────

def cosine_similarity(vec_a: dict, vec_b: dict) -> float:
    """
    Measure angular alignment between two TF-IDF vectors.
    cos(θ) = (A · B) / (||A|| * ||B||)
    Range: 0 (no match) → 1 (perfect match). Magnitude-invariant.
    """
    dot   = sum(vec_a.get(k, 0) * vec_b.get(k, 0) for k in vec_b)
    mag_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    mag_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0   # cold-start guard
    return dot / (mag_a * mag_b)


# ── Phase 3: 4-Step Ranking Pipeline ─────────────────────────────────────────

def recommend(user_skills: list, top_n: int = 3) -> list:
    """
    4-Step Pipeline:
      1. Ingestion  — build user profile vector from raw skill input
      2. Scoring    — cosine similarity of user vector vs every item vector
      3. Sorting    — rank all items descending by score
      4. Filtering  — return only Top-N to prevent choice overload
    """
    idf = compute_idf(CATALOGUE)

    # Step 1 — Ingestion
    user_tags   = [s.lower().strip() for s in user_skills]
    user_vector = vectorize(user_tags, idf)

    # Step 2 — Scoring
    scores = {}
    for role, tags in CATALOGUE.items():
        item_vector  = vectorize(tags, idf)
        scores[role] = cosine_similarity(user_vector, item_vector)

    # Step 3 — Sorting
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Step 4 — Filtering
    return ranked[:top_n]


# ── Phase 4: I/O Interface ────────────────────────────────────────────────────

def get_user_skills() -> list:
    """Onboarding survey — bypasses cold-start. Minimum 3 skills required."""
    print("\nEnter your skills one by one (min 3). Type 'done' when finished.")
    skills = []
    while True:
        skill = input(f"  Skill {len(skills)+1}: ").strip()
        if skill.lower() == "done":
            if len(skills) < 3:
                print("  Please enter at least 3 skills for accurate matching.")
            else:
                break
        elif skill:
            skills.append(skill)
    return skills


def display_results(results: list, user_skills: list) -> None:
    """Print ranked Top-N recommendations with visual score bar."""
    print("\n" + "=" * 52)
    print("  TOP CAREER RECOMMENDATIONS FOR YOUR PROFILE")
    print("=" * 52)
    print(f"  Skills: {', '.join(user_skills)}")
    print("-" * 52)
    for rank, (role, score) in enumerate(results, 1):
        bar = "█" * int(score * 20)
        print(f"  {rank}. {role}")
        print(f"     Match: {score:.2%}  {bar}")
    print("=" * 52)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 52)
    print("  DecodeLabs Tech Stack Recommender | P3")
    print("  Content-Based Filtering | TF-IDF + Cosine")
    print("=" * 52)

    user_skills = get_user_skills()
    results     = recommend(user_skills, top_n=3)
    display_results(results, user_skills)


if __name__ == "__main__":
    main()
