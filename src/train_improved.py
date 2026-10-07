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
# LOAD AUGMENTED TRAINING DATA
# ========================================

train_data = torch.load(
    "data/robustness/augmented_train_embeddings.pt"
)

X_train = train_data["embeddings"].numpy()
y_train = train_data["labels"].numpy()


# ========================================
# LOAD ORIGINAL TEST DATA
# ========================================

original_test = torch.load(
    "data/embeddings/test_embeddings.pt"
)

X_original_test = original_test["embeddings"].numpy()
y_original_test = original_test["labels"].numpy()


# ========================================
# LOAD UNSEEN ROBUSTNESS TEST DATA
# ========================================

robust_test = torch.load(
    "data/robustness/unseen_robustness_test_embeddings.pt"
)

X_robust_test = robust_test["embeddings"].numpy()
y_robust_test = robust_test["labels"].numpy()


print("========================================")
print("IMPROVED MODEL TRAINING")
print("========================================")

print("\nAugmented training data:")
print("Samples:", len(X_train))
print("Features:", X_train.shape[1])

print("\nOriginal test data:")
print("Samples:", len(X_original_test))

print("\nUnseen robustness test:")
print("Samples:", len(X_robust_test))


# ========================================
# DEFINE MODELS
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
# TRAIN AND EVALUATE
# ========================================

results = []


for name, model in models.items():

    print("\n========================================")
    print(name)
    print("========================================")

    # Train on augmented data
    model.fit(X_train, y_train)

    # ------------------------------------
    # ORIGINAL TEST
    # ------------------------------------

    original_predictions = model.predict(
        X_original_test
    )

    original_accuracy = accuracy_score(
        y_original_test,
        original_predictions
    )

    original_precision = precision_score(
        y_original_test,
        original_predictions,
        zero_division=0
    )

    original_recall = recall_score(
        y_original_test,
        original_predictions,
        zero_division=0
    )

    original_f1 = f1_score(
        y_original_test,
        original_predictions,
        zero_division=0
    )

    original_cm = confusion_matrix(
        y_original_test,
        original_predictions
    )


    # ------------------------------------
    # ROBUSTNESS TEST
    # ------------------------------------

    robust_predictions = model.predict(
        X_robust_test
    )

    robust_accuracy = accuracy_score(
        y_robust_test,
        robust_predictions
    )

    robust_precision = precision_score(
        y_robust_test,
        robust_predictions,
        zero_division=0
    )

    robust_recall = recall_score(
        y_robust_test,
        robust_predictions,
        zero_division=0
    )

    robust_f1 = f1_score(
        y_robust_test,
        robust_predictions,
        zero_division=0
    )

    robust_cm = confusion_matrix(
        y_robust_test,
        robust_predictions
    )


    # ------------------------------------
    # DISPLAY
    # ------------------------------------

    print("\nOriginal Test Set:")
    print(f"Accuracy : {original_accuracy:.4f}")
    print(f"Precision: {original_precision:.4f}")
    print(f"Recall   : {original_recall:.4f}")
    print(f"F1 Score : {original_f1:.4f}")

    print("Confusion Matrix:")
    print(original_cm)


    print("\nUnseen Robustness Test:")
    print(f"Accuracy : {robust_accuracy:.4f}")
    print(f"Precision: {robust_precision:.4f}")
    print(f"Recall   : {robust_recall:.4f}")
    print(f"F1 Score : {robust_f1:.4f}")

    print("Confusion Matrix:")
    print(robust_cm)


    # ------------------------------------
    # STORE RESULTS
    # ------------------------------------

    results.append({
        "Model": name,

        "Original_Test_Accuracy": original_accuracy,
        "Original_Test_Precision": original_precision,
        "Original_Test_Recall": original_recall,
        "Original_Test_F1": original_f1,

        "Robustness_Accuracy": robust_accuracy,
        "Robustness_Precision": robust_precision,
        "Robustness_Recall": robust_recall,
        "Robustness_F1": robust_f1
    })


# ========================================
# SAVE RESULTS
# ========================================

os.makedirs("results", exist_ok=True)

results_df = pd.DataFrame(results)

results_df.to_csv(
    "results/improved_results.csv",
    index=False
)


print("\n========================================")
print("IMPROVED MODEL RESULTS")
print("========================================")

print(results_df.to_string(index=False))

print("\nResults saved to:")
print("results/improved_results.csv")