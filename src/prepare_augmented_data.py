import pandas as pd
import torch


# ========================================
# LOAD ORIGINAL TRAINING EMBEDDINGS
# ========================================

original_train = torch.load(
    "data/embeddings/train_embeddings.pt"
)

original_embeddings = original_train["embeddings"]
original_labels = original_train["labels"]


# ========================================
# LOAD ROBUSTNESS EMBEDDINGS
# ========================================

robustness_data = torch.load(
    "data/robustness/robustness_embeddings.pt"
)

robustness_embeddings = robustness_data["embeddings"]
robustness_labels = robustness_data["labels"]


# ========================================
# LOAD SPLIT INFORMATION
# ========================================

train_df = pd.read_csv(
    "data/robustness/robustness_train.csv"
)

test_df = pd.read_csv(
    "data/robustness/robustness_test.csv"
)


# ========================================
# GET ROW INDICES
# ========================================

# Original robustness CSV has IDs 1-120.
# Convert IDs to zero-based embedding indices.

train_indices = train_df["id"].values - 1
test_indices = test_df["id"].values - 1


# ========================================
# SELECT ROBUSTNESS EMBEDDINGS
# ========================================

robust_train_embeddings = robustness_embeddings[
    train_indices
]

robust_train_labels = robustness_labels[
    train_indices
]

robust_test_embeddings = robustness_embeddings[
    test_indices
]

robust_test_labels = robustness_labels[
    test_indices
]


# ========================================
# CREATE AUGMENTED TRAINING SET
# ========================================

augmented_embeddings = torch.cat(
    [
        original_embeddings,
        robust_train_embeddings
    ],
    dim=0
)

augmented_labels = torch.cat(
    [
        original_labels,
        robust_train_labels
    ],
    dim=0
)


# ========================================
# SAVE AUGMENTED TRAINING DATA
# ========================================

torch.save(
    {
        "embeddings": augmented_embeddings,
        "labels": augmented_labels
    },
    "data/robustness/augmented_train_embeddings.pt"
)


# ========================================
# SAVE UNSEEN ROBUSTNESS TEST DATA
# ========================================

torch.save(
    {
        "embeddings": robust_test_embeddings,
        "labels": robust_test_labels
    },
    "data/robustness/unseen_robustness_test_embeddings.pt"
)


# ========================================
# DISPLAY RESULTS
# ========================================

print("========================================")
print("AUGMENTED DATA PREPARATION")
print("========================================")

print("\nOriginal training:")
print("Embeddings:", original_embeddings.shape)
print("Labels:", original_labels.shape)

print("\nRobustness augmentation:")
print("Embeddings:", robust_train_embeddings.shape)
print("Labels:", robust_train_labels.shape)

print("\nAugmented training:")
print("Embeddings:", augmented_embeddings.shape)
print("Labels:", augmented_labels.shape)

print("\nUnseen robustness test:")
print("Embeddings:", robust_test_embeddings.shape)
print("Labels:", robust_test_labels.shape)

print("\n========================================")
print("FILES SAVED")
print("========================================")

print("data/robustness/augmented_train_embeddings.pt")
print("data/robustness/unseen_robustness_test_embeddings.pt")