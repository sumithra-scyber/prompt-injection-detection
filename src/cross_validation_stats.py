import torch
import numpy as np

from sklearn.model_selection import StratifiedKFold

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from scipy.stats import friedmanchisquare, ttest_rel


# ============================================================
# Load training embeddings
# ============================================================

data = torch.load(
    "data/embeddings/train_embeddings.pt",
    weights_only=False
)

X = data["embeddings"].numpy()
y = data["labels"].numpy()


print("========================================")
print("10-FOLD CROSS-VALIDATION")
print("========================================")

print("Training samples:", X.shape[0])
print("Embedding dimension:", X.shape[1])


# ============================================================
# Define classifiers
# ============================================================

models = {

    "GNB": GaussianNB(),

    "RF": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),

    "SVM": SVC(
        kernel="rbf",
        random_state=42
    ),

    "LR": LogisticRegression(
        max_iter=1000,
        random_state=42
    )
}


# ============================================================
# 10-Fold Stratified Cross-Validation
# ============================================================

skf = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)


scores = {
    "GNB": [],
    "RF": [],
    "SVM": [],
    "LR": []
}


for fold, (train_idx, val_idx) in enumerate(
    skf.split(X, y),
    start=1
):

    X_train = X[train_idx]
    X_val = X[val_idx]

    y_train = y[train_idx]
    y_val = y[val_idx]


    print("\n========================================")
    print(f"Fold {fold}")
    print("========================================")


    for name, model in models.items():

        model.fit(
            X_train,
            y_train
        )

        accuracy = model.score(
            X_val,
            y_val
        )

        scores[name].append(accuracy)

        print(
            f"{name} : {accuracy:.4f}"
        )


# ============================================================
# Mean and Standard Deviation
# ============================================================

print("\n\n========================================")
print("CROSS-VALIDATION SUMMARY")
print("========================================")


for name in scores:

    mean_accuracy = np.mean(
        scores[name]
    )

    std_accuracy = np.std(
        scores[name],
        ddof=1
    )

    print(
        f"{name} → "
        f"Mean Accuracy = {mean_accuracy:.4f}, "
        f"Std = {std_accuracy:.4f}"
    )


# ============================================================
# Friedman Test
# ============================================================

friedman_stat, friedman_p = friedmanchisquare(
    scores["GNB"],
    scores["RF"],
    scores["SVM"],
    scores["LR"]
)


print("\n========================================")
print("FRIEDMAN TEST")
print("========================================")

print(
    f"Friedman statistic: {friedman_stat}"
)

print(
    f"Friedman p-value: {friedman_p}"
)


if friedman_p < 0.05:

    print(
        "Result: Significant difference "
        "among the algorithms."
    )

else:

    print(
        "Result: No significant difference "
        "among the algorithms."
    )


# ============================================================
# Pairwise Paired t-tests
# ============================================================

pairs = [

    ("GNB", "RF"),
    ("GNB", "SVM"),
    ("GNB", "LR"),
    ("RF", "SVM"),
    ("RF", "LR"),
    ("SVM", "LR")

]


p_values = []


print("\n========================================")
print("PAIRED T-TESTS")
print("========================================")


for model1, model2 in pairs:

    statistic, p_value = ttest_rel(
        scores[model1],
        scores[model2]
    )

    p_values.append(p_value)

    print(
        f"{model1} vs {model2} → "
        f"t = {statistic:.4f}, "
        f"p = {p_value:.6f}"
    )


# ============================================================
# Holm Correction
# ============================================================

print("\n========================================")
print("HOLM CORRECTION")
print("========================================")


alpha = 0.05

sorted_indices = np.argsort(
    p_values
)

adjusted_p_values = [
    0
] * len(p_values)

m = len(p_values)


for rank, index in enumerate(
    sorted_indices
):

    adjusted = (
        p_values[index]
        * (m - rank)
    )

    adjusted_p_values[index] = min(
        adjusted,
        1.0
    )


for i, (model1, model2) in enumerate(
    pairs
):

    adjusted_p = adjusted_p_values[i]

    if adjusted_p < alpha:

        result = "Significant"

    else:

        result = "Not Significant"


    print(
        f"{model1} vs {model2} → "
        f"Adjusted p = {adjusted_p:.6f} "
        f"→ {result}"
    )


# ============================================================
# Best Algorithm Based on Mean CV Accuracy
# ============================================================

mean_scores = {

    name: np.mean(scores[name])

    for name in scores
}


best_algorithm = max(
    mean_scores,
    key=mean_scores.get
)


print("\n========================================")
print("BEST ALGORITHM")
print("========================================")

print(
    f"Best algorithm based on mean CV accuracy: "
    f"{best_algorithm}"
)

print(
    f"Mean CV accuracy: "
    f"{mean_scores[best_algorithm]:.4f}"
)