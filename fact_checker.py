import chromadb
from ollama import chat
from textwrap import dedent

# === Connect to Chroma Vector DB ===
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection("pubmed_articles_mxbai")

# === Prompt Template ===
def build_prompt(claim, context_docs):
    joined_docs = "\n\n".join(context_docs)
    return dedent(f"""
        You are a scientific fact-checking assistant.

        Given the following claim:
        "{claim}"

        And the following evidence from scientific literature:
        {joined_docs}

        Determine if the claim is:
        - TRUE: if the evidence supports it
        - FALSE: if the evidence contradicts it
        - UNVERIFIABLE: if the evidence is insufficient or unrelated

        Respond with a clear verdict (TRUE, FALSE, or UNVERIFIABLE) followed by a short explanation based on the documents.
    """)

# === Main Loop ===
while True:
    claim = input("\n🔍 Enter a claim to fact-check (or 'exit' to quit):\n> ")
    if claim.lower() in {"exit", "quit"}:
        break

    # Retrieve top 5 most relevant documents
    results = collection.query(query_texts=[claim], n_results=5)
    docs = results["documents"][0]

    if not docs:
        print("⚠️ No evidence found in the database.")
        continue

    # Build prompt
    prompt = build_prompt(claim, docs)

    # Send to local LLM (Ollama must be running with llama3)
    response = chat(model="llama3", messages=[
        {"role": "user", "content": prompt}
    ])

    print("\n✅ Fact Check Result:")
    print(response["message"]["content"])
