import torch

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# Load saved embeddings
# ============================================================

train_data = torch.load(
    "data/embeddings/train_embeddings.pt",
    weights_only=True
)

test_data = torch.load(
    "data/embeddings/test_embeddings.pt",
    weights_only=True
)


# ============================================================
# Extract embeddings and labels
# ============================================================

X_train = train_data["embeddings"].numpy()
y_train = train_data["labels"].numpy()

X_test = test_data["embeddings"].numpy()
y_test = test_data["labels"].numpy()


print("Dataset loaded successfully!")

print("\nTraining data:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nTest data:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


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
# Train and evaluate models
# ============================================================

results = []


for name, model in models.items():

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model.fit(X_train, y_train)

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    y_pred = model.predict(X_test)

    # --------------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    # Store results
    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    })

    # --------------------------------------------------------
    # Print metrics
    # --------------------------------------------------------

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "Legitimate",
                "Malicious"
            ],
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    print("Confusion Matrix:")

    print(confusion_matrix(y_test, y_pred))


# ============================================================
# Final comparison
# ============================================================

print("\n")
print("=" * 80)
print("FINAL MODEL COMPARISON")
print("=" * 80)

print(
    f"{'Model':<25}"
    f"{'Accuracy':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1':<12}"
)

print("-" * 80)

for result in results:

    print(
        f"{result['Model']:<25}"
        f"{result['Accuracy']:<12.4f}"
        f"{result['Precision']:<12.4f}"
        f"{result['Recall']:<12.4f}"
        f"{result['F1']:<12.4f}"
    )

print("=" * 80)