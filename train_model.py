"""Step 1: Train a small CNN on MNIST.
Creates: digit_model.keras, training_curves.png, confusion_matrix.png, metrics.json
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

# 1. Load data (downloads automatically)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 2. Normalize to 0-1 and add channel dimension -> (28, 28, 1)
x_train = (x_train / 255.0).reshape(-1, 28, 28, 1)
x_test = (x_test / 255.0).reshape(-1, 28, 28, 1)

# 3. Build the CNN
model = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, 3, activation="relu"),   # detects edges/strokes
    layers.MaxPooling2D(),                     # shrinks the feature maps
    layers.Conv2D(64, 3, activation="relu"),   # detects shapes
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dropout(0.3),                       # reduces overfitting
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax"),    # 10 digit probabilities
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.summary()

# 4. Train
history = model.fit(x_train, y_train, epochs=8, batch_size=128,
                    validation_split=0.1)

# 5. Evaluate
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest accuracy: {acc:.4f}")

# 6. Save model
model.save("digit_model.keras")

# 7. Training curves
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(history.history["accuracy"], label="train")
ax[0].plot(history.history["val_accuracy"], label="validation")
ax[0].set_title("Accuracy"); ax[0].set_xlabel("Epoch"); ax[0].legend()
ax[1].plot(history.history["loss"], label="train")
ax[1].plot(history.history["val_loss"], label="validation")
ax[1].set_title("Loss"); ax[1].set_xlabel("Epoch"); ax[1].legend()
plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.close()

# 8. Confusion matrix (computed with numpy, no extra library)
y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
cm = np.zeros((10, 10), dtype=int)
np.add.at(cm, (y_test, y_pred), 1)

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(10)); ax.set_yticks(range(10))
ax.set_xlabel("Predicted digit"); ax.set_ylabel("Actual digit")
ax.set_title("Confusion Matrix (10,000 test images)")
for i in range(10):
    for j in range(10):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.colorbar(im)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()

# 9. Save metrics for the app
metrics = {
    "test_accuracy": float(acc),
    "test_loss": float(loss),
    "parameters": int(model.count_params()),
    "epochs": 8,
    "train_images": 60000,
    "test_images": 10000,
}
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Saved: digit_model.keras, training_curves.png, confusion_matrix.png, metrics.json")
