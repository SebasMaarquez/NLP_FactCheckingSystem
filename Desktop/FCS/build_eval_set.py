import chromadb
from transformers import AutoModel, AutoTokenizer
import torch

# Load embedding model 
model_name = "mixedbread-ai/mxbai-embed-large-v1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings[0].numpy().tolist()

# Connect to ChromaDB 
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection("pubmed_articles_mxbai")

# Define sample claims 
sample_claims = [
    "Alpha-bisabolol inhibits pepsin.",
    "Vitamin C cures the common cold.",
    "Smoking causes lung cancer.",
    "Aspirin prevents heart attacks.",
    "Penicillin treats bacterial infections.",
    "Omega-3 reduces inflammation.",
    "Antibiotics treat viral infections.",
    "Excessive sugar intake causes diabetes.",
    "Probiotics improve gut health.",
    "Air pollution increases asthma risk."
]

# Preview documents 
def explore_claims(claims, k=5):
    eval_set = []
    for claim in claims:
        print(f"\n\n=== Claim: {claim} ===")
        embedding = get_embedding(claim)
        results = collection.query(query_embeddings=[embedding], n_results=k, include=["documents"])
        top_docs = results["documents"][0]

        for i, doc in enumerate(top_docs):
            print(f"\nDoc {i+1} Preview:\n{doc[:500]}\n")

        # Optional: user input for labeling
        response = input("Enter exact matching sentence for this claim (or press Enter to skip):\n> ")
        if response.strip():
            eval_set.append({"claim": claim, "relevant_snippet": response.strip()})

    return eval_set

# Run and build evaluation set 
evaluation_set = explore_claims(sample_claims)
print("\nFinal Evaluation Set:")
for item in evaluation_set:
    print(item)

# Save to file 
import json
with open("evaluation_set.json", "w") as f:
    json.dump(evaluation_set, f, indent=2)
print("\nSaved as evaluation_set.json")
