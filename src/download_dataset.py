from datasets import load_dataset

dataset = load_dataset("deepset/prompt-injections")

'''print(dataset)'''

'''print("\nTraining columns:")
print(dataset["train"].column_names)

print("\nFirst training sample:")
print(dataset["train"][0])

print("\nFirst test sample:")
print(dataset["test"][0])'''

'''from collections import Counter

print("\nTraining label distribution:")
print(Counter(dataset["train"]["label"]))

print("\nTest label distribution:")
print(Counter(dataset["test"]["label"]))'''

'''print("\nLabel information:")
print(dataset["train"].features)

print("\nLabel feature:")
print(dataset["train"].features["label"])'''

'''print("\nMissing values:")
print("Train text:", sum(x is None for x in dataset["train"]["text"]))
print("Train label:", sum(x is None for x in dataset["train"]["label"]))

print("Test text:", sum(x is None for x in dataset["test"]["text"]))
print("Test label:", sum(x is None for x in dataset["test"]["label"]))'''

'''print("\nDuplicate texts:")

train_texts = dataset["train"]["text"]
test_texts = dataset["test"]["text"]

print("Duplicate rows in train:", len(train_texts) - len(set(train_texts)))
print("Duplicate rows in test:", len(test_texts) - len(set(test_texts)))

train_set = set(train_texts)
test_set = set(test_texts)

print("Train-test overlapping texts:", len(train_set & test_set))'''

'''print("\nEmpty or whitespace-only texts:")

train_empty = sum(not text.strip() for text in dataset["train"]["text"])
test_empty = sum(not text.strip() for text in dataset["test"]["text"])

print("Train:", train_empty)
print("Test:", test_empty)'''

print("\nText length statistics:")

train_lengths = [len(text.split()) for text in dataset["train"]["text"]]
test_lengths = [len(text.split()) for text in dataset["test"]["text"]]

print("Train:")
print("  Minimum:", min(train_lengths))
print("  Maximum:", max(train_lengths))
print("  Average:", sum(train_lengths) / len(train_lengths))

print("\nTest:")
print("  Minimum:", min(test_lengths))
print("  Maximum:", max(test_lengths))
print("  Average:", sum(test_lengths) / len(test_lengths))