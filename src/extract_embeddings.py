import os

import torch
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset


# ============================================================
# Load dataset
# ============================================================

dataset = load_dataset("deepset/prompt-injections")


# ============================================================
# Load multilingual BERT
# ============================================================

model_name = "bert-base-multilingual-cased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


# Use GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)
model.eval()

print("Using device:", device)


# ============================================================
# Mean pooling function
# ============================================================

def mean_pooling(model_output, attention_mask):

    # Shape:
    # [batch_size, sequence_length, 768]
    token_embeddings = model_output.last_hidden_state

    # Convert attention mask:
    # [batch_size, sequence_length]
    #
    # to:
    # [batch_size, sequence_length, 1]
    mask = attention_mask.unsqueeze(-1)

    # Ignore padding tokens
    masked_embeddings = token_embeddings * mask

    # Sum token embeddings
    sum_embeddings = masked_embeddings.sum(dim=1)

    # Count real tokens
    token_count = mask.sum(dim=1)

    # Calculate mean embedding
    mean_embeddings = sum_embeddings / token_count

    return mean_embeddings


# ============================================================
# Extract embeddings
# ============================================================

def extract_embeddings(texts, batch_size=16):

    all_embeddings = []

    for start in range(0, len(texts), batch_size):

        # Get current batch
        batch_texts = texts[start:start + batch_size]

        # ----------------------------------------------------
        # Tokenization
        # ----------------------------------------------------

        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        # Move tensors to CPU/GPU
        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        # ----------------------------------------------------
        # Generate mBERT representations
        # ----------------------------------------------------

        with torch.no_grad():
            outputs = model(**inputs)

        # ----------------------------------------------------
        # Mean pooling
        # ----------------------------------------------------

        embeddings = mean_pooling(
            outputs,
            inputs["attention_mask"]
        )

        # Move embeddings back to CPU
        all_embeddings.append(embeddings.cpu())

        # Progress message
        processed = min(start + batch_size, len(texts))

        print(
            f"Processed {processed}/{len(texts)}"
        )

    # Combine all batches
    return torch.cat(all_embeddings, dim=0)


# ============================================================
# Training embeddings
# ============================================================

train_texts = dataset["train"]["text"]
train_labels = dataset["train"]["label"]

print("\nExtracting training embeddings...")

train_embeddings = extract_embeddings(train_texts)

print("\nTraining embeddings shape:")
print(train_embeddings.shape)


# ============================================================
# Test embeddings
# ============================================================

test_texts = dataset["test"]["text"]
test_labels = dataset["test"]["label"]

print("\nExtracting test embeddings...")

test_embeddings = extract_embeddings(test_texts)

print("\nTest embeddings shape:")
print(test_embeddings.shape)


# ============================================================
# Save embeddings and labels
# ============================================================

os.makedirs("data/embeddings", exist_ok=True)


# Save training embeddings
torch.save(
    {
        "embeddings": train_embeddings,
        "labels": torch.tensor(train_labels)
    },
    "data/embeddings/train_embeddings.pt"
)


# Save test embeddings
torch.save(
    {
        "embeddings": test_embeddings,
        "labels": torch.tensor(test_labels)
    },
    "data/embeddings/test_embeddings.pt"
)


# ============================================================
# Final verification
# ============================================================

print("\n========================================")
print("Embeddings saved successfully!")
print("========================================")

print("\nTraining:")
print("  Embeddings:", train_embeddings.shape)
print("  Labels:", len(train_labels))

print("\nTest:")
print("  Embeddings:", test_embeddings.shape)
print("  Labels:", len(test_labels))

print("\nSaved files:")
print("  data/embeddings/train_embeddings1.pt")
print("  data/embeddings/test_embeddings1.pt")