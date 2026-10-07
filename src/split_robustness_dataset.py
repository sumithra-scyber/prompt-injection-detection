import pandas as pd
from sklearn.model_selection import train_test_split


# ========================================
# LOAD ROBUSTNESS DATASET
# ========================================

df = pd.read_csv(
    "data/robustness/indirect_prompt_injections.csv"
)

print("========================================")
print("SPLITTING ROBUSTNESS DATASET")
print("========================================")

print("Total samples:", len(df))
print("Benign:", (df["label"] == 0).sum())
print("Malicious:", (df["label"] == 1).sum())


# ========================================
# STRATIFIED SPLIT
# ========================================

train_df, test_df = train_test_split(
    df,
    test_size=40,
    stratify=df["label"],
    random_state=42
)


# ========================================
# SAVE
# ========================================

train_df = train_df.sort_values("id")
test_df = test_df.sort_values("id")

train_df.to_csv(
    "data/robustness/robustness_train.csv",
    index=False
)

test_df.to_csv(
    "data/robustness/robustness_test.csv",
    index=False
)


# ========================================
# DISPLAY RESULTS
# ========================================

print("\n========================================")
print("ROBUSTNESS TRAINING SET")
print("========================================")

print("Samples:", len(train_df))
print("Benign:", (train_df["label"] == 0).sum())
print("Malicious:", (train_df["label"] == 1).sum())

print("\nCategory distribution:")
print(train_df["category"].value_counts())


print("\n========================================")
print("ROBUSTNESS TEST SET")
print("========================================")

print("Samples:", len(test_df))
print("Benign:", (test_df["label"] == 0).sum())
print("Malicious:", (test_df["label"] == 1).sum())

print("\nCategory distribution:")
print(test_df["category"].value_counts())


print("\n========================================")
print("FILES SAVED")
print("========================================")

print("data/robustness/robustness_train.csv")
print("data/robustness/robustness_test.csv")