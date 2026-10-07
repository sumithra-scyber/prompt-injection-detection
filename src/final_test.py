import os

import torch
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# Load training embeddings
# ============================================================

train_data = torch.load(
    "data/embeddings/train_embeddings.pt",
    weights_only=False
)

X_train = train_data["embeddings"].numpy()
y_train = train_data["labels"].numpy()


# ============================================================
# Load test embeddings
# ============================================================

test_data = torch.load(
    "data/embeddings/test_embeddings.pt",
    weights_only=False
)

X_test = test_data["embeddings"].numpy()
y_test = test_data["labels"].numpy()


print("========================================")
print("FINAL TESTING")
print("========================================")

print("Training samples:", X_train.shape[0])
print("Test samples:", X_test.shape[0])
print("Embedding dimension:", X_train.shape[1])


# ============================================================
# Define classifiers
# ============================================================

models = {
    "Gaussian Naive Bayes": GaussianNB(),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),

    "SVM": SVC(
        kernel="rbf",
        random_state=42
    ),

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    )
}


# ============================================================
# Train and evaluate
# ============================================================

results = []


for name, model in models.items():

    print("\n========================================")
    print(name)
    print("========================================")

    # Train using ALL 546 training samples
    model.fit(X_train, y_train)

    # Predict on untouched 116 test samples
    predictions = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    cm = confusion_matrix(
        y_test,
        predictions
    )

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Legitimate", "Malicious"],
            zero_division=0
        )
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    })


# ============================================================
# Results DataFrame
# ============================================================

results_df = pd.DataFrame(results)

print("\n========================================")
print("FINAL TEST RESULTS")
print("========================================")

print(
    results_df.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.4f}".format,
            "Precision": "{:.4f}".format,
            "Recall": "{:.4f}".format,
            "F1": "{:.4f}".format
        }
    )
)


# ============================================================
# Save results
# ============================================================

os.makedirs("results", exist_ok=True)

results_df.to_csv(
    "results/final_test_results.csv",
    index=False
)

print("\nResults saved to:")
print("results/final_test_results.csv")


# ============================================================
# Accuracy Graph
# ============================================================

plt.figure(figsize=(9, 6))

plt.bar(
    results_df["Model"],
    results_df["Accuracy"] * 100
)

plt.xlabel("Classification Algorithm")
plt.ylabel("Accuracy (%)")
plt.title("Final Test Accuracy of Classification Algorithms")

plt.ylim(0, 100)

plt.xticks(
    rotation=20,
    ha="right"
)

# Display accuracy values above bars
for i, accuracy in enumerate(results_df["Accuracy"]):

    plt.text(
        i,
        accuracy * 100 + 1,
        f"{accuracy * 100:.2f}%",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    "results/final_test_accuracy.png",
    dpi=300
)

plt.show()

print("\nAccuracy graph saved to:")
print("results/final_test_accuracy.png")