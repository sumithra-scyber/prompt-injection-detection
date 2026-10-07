import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel


# ========================================
# LOAD ROBUSTNESS DATASET
# ========================================

df = pd.read_csv(
    "data/robustness/indirect_prompt_injections.csv"
)

texts = df["text"].tolist()
labels = df["label"].tolist()

print("========================================")
print("ROBUSTNESS EMBEDDING EXTRACTION")
print("========================================")
print("Total samples:", len(texts))
print("Benign samples:", labels.count(0))
print("Malicious samples:", labels.count(1))


# ========================================
# LOAD mBERT
# ========================================

model_name = "bert-base-multilingual-cased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model.to(device)
model.eval()

print("Using device:", device)


# ========================================
# MEAN POOLING
# ========================================

def mean_pooling(model_output, attention_mask):

    token_embeddings = model_output.last_hidden_state

    mask = attention_mask.unsqueeze(-1)

    masked_embeddings = token_embeddings * mask

    sum_embeddings = masked_embeddings.sum(dim=1)

    token_count = mask.sum(dim=1)

    mean_embeddings = sum_embeddings / token_count

    return mean_embeddings


# ========================================
# EXTRACT EMBEDDINGS
# ========================================

def extract_embeddings(texts, batch_size=16):

    all_embeddings = []

    for start in range(0, len(texts), batch_size):

        batch_texts = texts[start:start + batch_size]

        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        with torch.no_grad():

            outputs = model(**inputs)

        embeddings = mean_pooling(
            outputs,
            inputs["attention_mask"]
        )

        all_embeddings.append(
            embeddings.cpu()
        )

        processed = min(
            start + batch_size,
            len(texts)
        )

        print(
            f"Processed {processed}/{len(texts)}"
        )

    return torch.cat(
        all_embeddings,
        dim=0
    )


# ========================================
# GENERATE EMBEDDINGS
# ========================================

print("\nExtracting robustness embeddings...")

robustness_embeddings = extract_embeddings(texts)

print("\nRobustness embeddings shape:")
print(robustness_embeddings.shape)


# ========================================
# SAVE
# ========================================

os.makedirs(
    "data/robustness",
    exist_ok=True
)

torch.save(
    {
        "embeddings": robustness_embeddings,
        "labels": torch.tensor(labels)
    },
    "data/robustness/robustness_embeddings.pt"
)


print("\n========================================")
print("EMBEDDINGS SAVED SUCCESSFULLY!")
print("========================================")

print(
    "Embeddings:",
    robustness_embeddings.shape
)

print(
    "Labels:",
    len(labels)
)

print(
    "\nSaved to:"
)

print(
    "data/robustness/robustness_embeddings.pt"
)