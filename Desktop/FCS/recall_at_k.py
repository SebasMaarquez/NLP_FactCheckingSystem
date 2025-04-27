import json
import chromadb
from transformers import AutoModel, AutoTokenizer
import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load evaluation set
with open("evaluation_set.json") as f:
    evaluation_set = json.load(f)

# Load embedding model (must match the one used for indexing)
model_name = "mixedbread-ai/mxbai-embed-large-v1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings[0].numpy()

# Connect to vector DB
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection("pubmed_articles_mxbai")

ks = [1, 3, 5, 10]
recall_at_k = {k: 0 for k in ks}

SIM_THRESHOLD = 0.75

for item in tqdm(evaluation_set, desc="Evaluating Recall@K"):
    claim = item["claim"]
    reference_text = item.get("relevant_snippet") or item.get("relevant_sentence") or item.get("ground_truth_snippet")

    if not reference_text:
        print(f"Skipping claim (missing reference): {claim}")
        continue

    # Embed the claim and the reference
    claim_embedding = get_embedding(claim)
    reference_embedding = get_embedding(reference_text).reshape(1, -1)

    # Query DB
    results = collection.query(query_embeddings=[claim_embedding.tolist()], n_results=max(ks), include=["documents"])
    docs = results["documents"][0]

    # Get document embeddings
    doc_embeddings = [get_embedding(doc) for doc in docs]
    doc_embeddings_matrix = np.stack(doc_embeddings)

    # Cosine similarity with reference
    sims = cosine_similarity(reference_embedding, doc_embeddings_matrix)[0]

    for k in ks:
        if np.any(sims[:k] >= SIM_THRESHOLD):
            recall_at_k[k] += 1

# Final results
print("\nRecall@K Results:")
total = len(evaluation_set)
for k in ks:
    recall = recall_at_k[k] / total
    print(f"Recall@{k}: {recall:.2f}")
