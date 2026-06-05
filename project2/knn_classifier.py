# ============================================================
#  Project 2 – Data Classification Using AI (KNN)
#  DecodeLabs Industrial Training | Batch 2026
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (confusion_matrix, f1_score,
                             classification_report, accuracy_score)

# ── 1. LOAD & UNDERSTAND DATASET ─────────────────────────────
iris = load_iris()
X, y = iris.data, iris.target
print("=" * 50)
print("  PROJECT 2: Data Classification Using KNN")
print("=" * 50)
print(f"\n[Dataset Info]")
print(f"  Total Samples  : {X.shape[0]}")
print(f"  Features       : {X.shape[1]} {list(iris.feature_names)}")
print(f"  Classes        : {iris.target_names.tolist()}")

# ── 2. TRAIN-TEST SPLIT ───────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True)
print(f"\n[Split]  Train: {len(X_train)} | Test: {len(X_test)}")

# ── 3. FEATURE SCALING ────────────────────────────────────────
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ── 4. FIND OPTIMAL K ────────────────────────────────────────
error_rates = []
for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    error_rates.append(1 - accuracy_score(y_test, knn.predict(X_test)))

optimal_k = error_rates.index(min(error_rates)) + 1
print(f"\n[Optimal K] = {optimal_k}")

# ── 5. TRAIN FINAL MODEL ──────────────────────────────────────
model = KNeighborsClassifier(n_neighbors=optimal_k)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# ── 6. RESULTS ───────────────────────────────────────────────
acc = accuracy_score(y_test, predictions)
f1  = f1_score(y_test, predictions, average='weighted')
cm  = confusion_matrix(y_test, predictions)

print(f"\n[Results]")
print(f"  Accuracy : {acc:.4f}")
print(f"  F1 Score : {f1:.4f}")
print(f"\n[Confusion Matrix]\n{cm}")
print(f"\n[Classification Report]\n"
      f"{classification_report(y_test, predictions, target_names=iris.target_names)}")

# ── 7. SAVE SCREENSHOTS ──────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Project 2 – KNN Data Classification | DecodeLabs 2026",
             fontsize=14, fontweight='bold')

# Plot 1: Elbow Curve
axes[0].plot(range(1, 21), error_rates, marker='o', color='steelblue', linewidth=2)
axes[0].axvline(x=optimal_k, color='red', linestyle='--', label=f'Optimal K={optimal_k}')
axes[0].set_title("Elbow Curve – Choosing Optimal K")
axes[0].set_xlabel("K Value"); axes[0].set_ylabel("Error Rate")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

# Plot 2: Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
            xticklabels=iris.target_names, yticklabels=iris.target_names)
axes[1].set_title(f"Confusion Matrix (K={optimal_k})")
axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Actual")

# Plot 3: Feature Distribution
colors = ['#E74C3C', '#3498DB', '#2ECC71']
for i, (cls, col) in enumerate(zip(iris.target_names, colors)):
    mask = y == i
    axes[2].scatter(X[mask, 2], X[mask, 3], c=col, label=cls, alpha=0.7, s=60)
axes[2].set_title("Petal Length vs Petal Width")
axes[2].set_xlabel("Petal Length (cm)"); axes[2].set_ylabel("Petal Width (cm)")
axes[2].legend(); axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("screenshots/results.png", dpi=150, bbox_inches='tight')
plt.close()
print("\n[Screenshot saved] → screenshots/results.png")
