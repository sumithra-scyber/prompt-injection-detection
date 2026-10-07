from transformers import AutoTokenizer
from datasets import load_dataset

# Load dataset
dataset = load_dataset("deepset/prompt-injections")

# Load multilingual BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-multilingual-cased"
)

print("Tokenizer loaded successfully!")
print("Maximum length:", tokenizer.model_max_length)

# Combine train and test prompts
texts = list(dataset["train"]["text"]) + list(dataset["test"]["text"])

# Get token lengths without truncation
token_lengths = []

for text in texts:
    tokens = tokenizer(
        text,
        add_special_tokens=True,
        truncation=False
    )

    token_lengths.append(len(tokens["input_ids"]))

print("\nToken length statistics:")
print("Minimum:", min(token_lengths))
print("Maximum:", max(token_lengths))
print("Average:", sum(token_lengths) / len(token_lengths))

# Count prompts exceeding 512 tokens
over_512 = sum(length > 512 for length in token_lengths)

print("\nPrompts exceeding 512 tokens:", over_512)
print(
    "Percentage exceeding 512:",
    (over_512 / len(token_lengths)) * 100,
    "%"
)