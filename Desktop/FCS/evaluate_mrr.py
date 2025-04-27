import json
import chromadb
from transformers import AutoModel, AutoTokenizer
import torch
from tqdm import tqdm
import re
from ollama import chat

# Load evaluation set
with open("evaluation_set.json") as f:
    evaluation_set = json.load(f)

print(f"Loaded {len(evaluation_set)} items for evaluation.")

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

# Connect to vector DB
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection("pubmed_articles_mxbai")

reciprocal_ranks = []

for item in tqdm(evaluation_set, desc="Evaluating MRR with LLM reranking"):
    claim = item["claim"]
    reference = item["relevant_snippet"]

    embedding = get_embedding(claim)
    results = collection.query(query_embeddings=[embedding], n_results=5)
    docs = results["documents"][0]

    # Construct prompt to ask LLM to pick most relevant document
    joined_docs = "\n\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(docs)])
    prompt = f"""
    Based on the claim below, identify the number of the document that is most relevant.
    Claim: "{claim}"

    Documents:
    {joined_docs}

    Respond with the document number only (e.g., '2').
    """

    try:
        response = chat(model="llama3", messages=[{"role": "user", "content": prompt}])
        answer = response["message"]["content"]

        match = re.search(r"\b(\d)\b", answer)
        if match:
            predicted_index = int(match.group(1)) - 1
            if predicted_index < len(docs):
                if reference.lower() in docs[predicted_index].lower():
                    reciprocal_ranks.append(1 / (predicted_index + 1))
                else:
                    # If the predicted doc doesn't contain reference, search for it manually
                    found = False
                    for rank, doc in enumerate(docs):
                        if reference.lower() in doc.lower():
                            reciprocal_ranks.append(1 / (rank + 1))
                            found = True
                            break
                    if not found:
                        reciprocal_ranks.append(0)
            else:
                reciprocal_ranks.append(0)
        else:
            print(f"Could not parse number from response: {answer.strip()}")
            reciprocal_ranks.append(0)

    except Exception as e:
        print(f"Error for claim: {claim}\n{e}")
        reciprocal_ranks.append(0)

# Final score
mrr = sum(reciprocal_ranks) / len(evaluation_set)
print(f"\nMean Reciprocal Rank (LLM Reranked): {mrr:.2f}")