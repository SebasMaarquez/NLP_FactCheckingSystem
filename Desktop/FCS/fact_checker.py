import chromadb
from ollama import chat
from textwrap import dedent
from transformers import AutoModel, AutoTokenizer
import torch
from langdetect import detect
import langcodes

# Load the same embedding model used during indexing 
model_name = "mixedbread-ai/mxbai-embed-large-v1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings[0].numpy().tolist()

# Connect to Chroma Vector DB 
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection("pubmed_articles_mxbai")

# Prompt Template
def build_prompt(claim, context_docs):
    language_code = detect(claim)
    lang_name = langcodes.Language.make(language_code).display_name("en") or "the input language"

    joined_docs = "\n\n".join(context_docs)
    return dedent(f"""
        IMPORTANT: Respond ONLY in {lang_name}. Do not use any other language.

        You are a scientific fact-checking assistant.

        Given the following claim:
        "{claim}"

        And the following evidence from scientific literature:
        {joined_docs}

        Determine if the claim is:
        - TRUE: if the evidence supports it
        - FALSE: if the evidence contradicts it
        - UNVERIFIABLE: if the evidence is insufficient or unrelated

        Only use the evidence provided. Do not assume facts not in the documents. Do not speculate.

        Your response must begin with the verdict (TRUE, FALSE, or UNVERIFIABLE) followed by a short explanation based only on the provided evidence.
    """)

# Main Loop 
while True:
    claim = input("\n🔍 Enter a claim to fact-check (or 'exit' to quit):\n> ")
    if claim.lower() in {"exit", "quit"}:
        break

    # Embed the claim using mxbai model
    embedding = get_embedding(claim)

    # Query with the correct dimension
    results = collection.query(query_embeddings=[embedding], n_results=5)
    docs = results["documents"][0]

    if not docs:
        print("⚠️ No evidence found in the database.")
        continue

    # Build and send the prompt
    prompt = build_prompt(claim, docs)

    response = chat(model="llama3", messages=[
        {"role": "user", "content": prompt}
    ])

    print("\n✅ Fact Check Result:")
    print(response["message"]["content"])