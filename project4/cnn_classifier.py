# ============================================================
#  Project 4 – Image Classification Using CNN (MNIST)
#  DecodeLabs Industrial Training | Batch 2026
#  ▶  Run on Google Colab for GPU acceleration
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

print("=" * 50)
print("  PROJECT 4: Image Classification Using CNN")
print("=" * 50)
print(f"\n[TensorFlow version] {tf.__version__}")

# ── 1. LOAD & PREPROCESS ─────────────────────────────────────
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
X_test  = X_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

print(f"\n[Dataset Info]")
print(f"  Train : {X_train.shape}  |  Labels: {y_train.shape}")
print(f"  Test  : {X_test.shape}   |  Labels: {y_test.shape}")
print(f"  Classes: {np.unique(y_train).tolist()} (digits 0–9)")

# ── 2. BUILD CNN ─────────────────────────────────────────────
model = models.Sequential([
    # Conv Block 1 – edge/low-level feature detection
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),

    # Conv Block 2 – complex feature detection
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Classifier Head
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),            # Prevents overfitting
    layers.Dense(10, activation='softmax')
], name="MNIST_CNN")

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# ── 3. TRAIN ─────────────────────────────────────────────────
print("\n[Training...]")
history = model.fit(X_train, y_train,
                    epochs=5,
                    batch_size=64,
                    validation_split=0.1,
                    verbose=1)

# ── 4. EVALUATE ──────────────────────────────────────────────
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n[Results]")
print(f"  Test Accuracy : {test_acc:.4f}")
print(f"  Test Loss     : {test_loss:.4f}")

# ── 5. METRICS ───────────────────────────────────────────────
y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
cm = confusion_matrix(y_test, y_pred)
print(f"\n[Confusion Matrix]\n{cm}")
print(f"\n[Classification Report]\n"
      f"{classification_report(y_test, y_pred, target_names=[str(i) for i in range(10)])}")

# ── 6. SAVE SCREENSHOTS ──────────────────────────────────────
import os
os.makedirs("screenshots", exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Project 4 – CNN Image Classification | DecodeLabs 2026",
             fontsize=14, fontweight='bold')

# Plot 1: Training curves
axes[0].plot(history.history['accuracy'],     label='Train Acc',  color='steelblue')
axes[0].plot(history.history['val_accuracy'], label='Val Acc',    color='orange')
axes[0].plot(history.history['loss'],         label='Train Loss', color='steelblue', linestyle='--')
axes[0].plot(history.history['val_loss'],     label='Val Loss',   color='orange',    linestyle='--')
axes[0].set_title("Training History")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Value")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

# Plot 2: Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
            xticklabels=range(10), yticklabels=range(10))
axes[1].set_title(f"Confusion Matrix\n(Acc={test_acc:.4f})")
axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Actual")

# Plot 3: Sample predictions
for i in range(10):
    ax = fig.add_subplot(2, 5, i + 1) if i == 0 else None  # just placeholder
axes[2].axis('off')
sample_indices = np.random.choice(len(X_test), 25, replace=False)
grid_img = np.zeros((5 * 28, 5 * 28))
for idx, si in enumerate(sample_indices[:25]):
    r, c = divmod(idx, 5)
    grid_img[r*28:(r+1)*28, c*28:(c+1)*28] = X_test[si].reshape(28, 28)
axes[2].imshow(grid_img, cmap='gray')
axes[2].set_title("Sample Test Images")
axes[2].axis('off')

plt.tight_layout()
plt.savefig("screenshots/results.png", dpi=150, bbox_inches='tight')
plt.close()
print("\n[Screenshot saved] → screenshots/results.png")

# Save model
model.save("cnn_mnist_model.h5")
print("[Model saved]  → cnn_mnist_model.h5")
