import streamlit as st
import chromadb
from ollama import chat
from transformers import AutoModel, AutoTokenizer
import torch
from langdetect import detect
import langcodes

# Setup page 
st.set_page_config(page_title="Scientific Fact Checker", page_icon="🧠")
st.title("🧠 Scientific Fact Checker")
st.write("Enter a scientific claim and get a verdict based on PubMed evidence")

# Claim input 
claim = st.text_input("🔍 Enter a claim to fact-check:", placeholder="e.g. Alpha-bisabolol inhibits pepsin.")

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

# === Connect to ChromaDB ===
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection("pubmed_articles_mxbai")

# === Retrieve Documents ===
def retrieve_documents(claim, k=3):
    embedding = get_embedding(claim)
    results = collection.query(query_embeddings=[embedding], n_results=k)
    return results["documents"][0]

# === LLM Prompt Construction ===
def build_prompt(claim, context_docs):
    language_code = detect(claim)
    lang_name = langcodes.Language.make(language_code).display_name("en") or "the input language"
    joined_docs = "\n\n".join(context_docs)

    return f"""
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
    """

# === Get Verdict ===
def get_llm_verdict(prompt):
    response = chat(model="llama3", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

def parse_verdict(text):
    text = text.strip()
    if text.upper().startswith("TRUE"):
        return "TRUE", text
    elif text.upper().startswith("FALSE"):
        return "FALSE", text
    elif text.upper().startswith("UNVERIFIABLE"):
        return "UNVERIFIABLE", text
    else:
        return "UNKNOWN", text

# === Fact Check Button ===
if st.button("🦖 Fact-Check Claim") and claim:
    with st.spinner("Analyzing claim and retrieving evidence..."):
        docs = retrieve_documents(claim)
        prompt = build_prompt(claim, docs)
        llm_response = get_llm_verdict(prompt)
        verdict, explanation = parse_verdict(llm_response)

    # === Display user input ===
    st.markdown("#### 👨‍💼 **User Claim:**")
    st.info(claim)

    # === Display verdict ===
    if verdict == "TRUE":
        st.success(f"**System Verdict: {verdict}**\n\n{explanation}")
    elif verdict == "FALSE":
        st.error(f"**System Verdict: {verdict}**\n\n{explanation}")
    elif verdict == "UNVERIFIABLE":
        st.warning(f"**System Verdict: {verdict}**\n\n{explanation}")
    else:
        st.info(f"**System Verdict:**\n\n{explanation}")

    # === Show retrieved evidence ===
    with st.expander("🔍 Show Retrieved Evidence"):
        for i, doc in enumerate(docs):
            st.markdown(f"**Document {i+1}:** {doc[:500]}...")