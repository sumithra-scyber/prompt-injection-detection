import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.svm import SVC


# ========================================
# LOAD mBERT
# ========================================

model_name = "bert-base-multilingual-cased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
bert_model = AutoModel.from_pretrained(model_name)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

bert_model.to(device)
bert_model.eval()


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
# LOAD AUGMENTED TRAINING DATA
# ========================================

train_data = torch.load(
    "data/robustness/augmented_train_embeddings.pt"
)

X_train = train_data["embeddings"].numpy()
y_train = train_data["labels"].numpy()


# ========================================
# TRAIN IMPROVED SVM
# ========================================

classifier = SVC(
    kernel="rbf",
    probability=True,
    random_state=42
)

classifier.fit(X_train, y_train)


# ========================================
# FUNCTION TO GENERATE EMBEDDING
# ========================================

def get_embedding(text):

    inputs = tokenizer(
        text,
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

        outputs = bert_model(**inputs)

    embedding = mean_pooling(
        outputs,
        inputs["attention_mask"]
    )

    return embedding.cpu().numpy()


# ========================================
# PREDICT FUNCTION
# ========================================

def predict_prompt(text):

    embedding = get_embedding(text)

    prediction = classifier.predict(embedding)[0]

    probabilities = classifier.predict_proba(embedding)[0]

    confidence = probabilities[prediction]

    if prediction == 1:

        label = "MALICIOUS"

    else:

        label = "BENIGN"

    return label, confidence


# ========================================
# INTERACTIVE DETECTOR
# ========================================

print("========================================")
print("PROMPT INJECTION DETECTOR")
print("========================================")

print("Model: mBERT + SVM")
print("Training samples: 626")
print("Enter 'exit' to quit.")

while True:

    print()

    prompt = input("Enter prompt: ")

    if prompt.lower() == "exit":
        print("\nExiting detector...")
        break

    if not prompt.strip():

        print("Please enter a prompt.")

        continue

    label, confidence = predict_prompt(prompt)

    print("\nPrediction:", label)

    print(
        f"Confidence: {confidence * 100:.2f}%"
    )