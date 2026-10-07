import os
import torch
import pandas as pd

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ========================================
# LOAD ORIGINAL TRAINING DATA
# ========================================

train_data = torch.load(
    "data/embeddings/train_embeddings.pt",
    weights_only=False
)

X_train = train_data["embeddings"].numpy()
y_train = train_data["labels"].numpy()


# ========================================
# LOAD ROBUSTNESS DATA
# ========================================

robustness_data = torch.load(
    "data/robustness/robustness_embeddings.pt",
    weights_only=False
)

X_robust = robustness_data["embeddings"].numpy()
y_robust = robustness_data["labels"].numpy()


print("========================================")
print("ROBUSTNESS TESTING")
print("========================================")
print("Original training samples:", X_train.shape[0])
print("Robustness samples:", X_robust.shape[0])
print("Embedding dimension:", X_train.shape[1])


# ========================================
# MODELS
# ========================================

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


# ========================================
# ROBUSTNESS EVALUATION
# ========================================

results = []

for name, model in models.items():

    print("\n========================================")
    print(name)
    print("========================================")

    # Train ONLY on original training data
    model.fit(X_train, y_train)

    # Predict on NEW robustness dataset
    predictions = model.predict(X_robust)

    accuracy = accuracy_score(
        y_robust,
        predictions
    )

    precision = precision_score(
        y_robust,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_robust,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_robust,
        predictions,
        zero_division=0
    )

    cm = confusion_matrix(
        y_robust,
        predictions
    )

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    })


# ========================================
# RESULTS TABLE
# ========================================

results_df = pd.DataFrame(results)

print("\n\n========================================")
print("ROBUSTNESS RESULTS")
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


# ========================================
# SAVE RESULTS
# ========================================

os.makedirs(
    "results",
    exist_ok=True
)

results_df.to_csv(
    "results/robustness_results.csv",
    index=False
)

print("\nResults saved to:")
print("results/robustness_results.csv")