import os
import torch
from transformers import AutoModel, AutoTokenizer
import chromadb
from parse_pubmed import parse_pubmed_file
from tqdm import tqdm

# === Constants ===
DB_DIR = "chroma_db"
COLLECTION_NAME = "pubmed_articles_mxbai"
DATA_FILE = "pubmed25n0001.xml.gz"

# === Connect to Persistent Chroma DB ===
client = chromadb.PersistentClient(path=DB_DIR)

# Check if the collection already exists
existing_collections = [c.name for c in client.list_collections()]
if COLLECTION_NAME in existing_collections:
    print(f"Collection '{COLLECTION_NAME}' already exists. Loading without re-indexing.")
    collection = client.get_or_create_collection(COLLECTION_NAME)

else:
    print(f"Collection not found. Starting indexing process...")

    # === Parse articles ===
    articles = parse_pubmed_file(DATA_FILE)
    articles = articles[:3500]

    # === Load mxbai-embed-large model ===
    print("Loading mxbai-embed-large model...")
    model_name = "mixedbread-ai/mxbai-embed-large-v1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    def get_embedding(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings[0].numpy().tolist()

    # === Create collection ===
    collection = client.get_or_create_collection(COLLECTION_NAME)

 # === Index with progress bar ===
for i, article in tqdm(enumerate(articles), total=len(articles)):
    title = article['title'].strip()
    abstract = article['abstract'].strip()
    full_text = f"{title}. {abstract}"

    embedding = get_embedding(full_text)

    collection.add(
        documents=[full_text],  # used for search
        embeddings=[embedding],
        ids=[f"doc_{i}"],
        metadatas=[{
            "title": title,
            "abstract": abstract
        }]
    )

print(f"Indexed {len(articles)} articles into '{COLLECTION_NAME}'.")
