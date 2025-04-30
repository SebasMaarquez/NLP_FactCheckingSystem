import streamlit as st
import chromadb
from ollama import chat
from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np
import json
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity

# Setup page 
st.set_page_config(page_title="Scientific Fact Checker", page_icon="🧠")
st.title("🧠 Scientific Fact Checker")
st.write("Enter a scientific claim and get a verdict based on PubMed evidence")

# Claim Input 
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

# Connect to ChromaDB 
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection("pubmed_articles_mxbai")

def retrieve_documents(claim, k=3):
    embedding = get_embedding(claim)
    results = collection.query(query_embeddings=[embedding], n_results=k, include=["documents", "distances"])
    docs = results["documents"][0]
    distances = results["distances"][0]

    titles = []
    best_fragments = []

    for doc in docs:
        if ". " in doc:
            title, abstract = doc.split(". ", 1)
        else:
            title, abstract = "Untitled", doc

        sentences = sent_tokenize(abstract)
        sentence_embeddings = [get_embedding(s) for s in sentences]
        similarities = cosine_similarity([embedding], sentence_embeddings)[0]
        best_idx = int(np.argmax(similarities))

        titles.append(title.strip())
        best_fragments.append(sentences[best_idx].strip())

    return docs, titles, best_fragments, distances

# Prompt construction
def build_prompt(claim, context_docs):
    doc_blocks = []
    for i, doc in enumerate(context_docs):
        if ". " in doc:
            title, abstract = doc.split(". ", 1)
        else:
            title, abstract = "Untitled", doc
        doc_blocks.append(f"Document {i+1}:\n{abstract.strip()}")
    joined_docs = "\n\n".join(doc_blocks)

    return f"""
You are a multilingual scientific fact-checking assistant.

Here is a user's claim:
"{claim}"

And here is the evidence from biomedical literature:
{joined_docs}

Your task:
1. Detect the language of the claim and provide your response in that language.
2. Determine the correct verdict:
   - "TRUE" if at least one document clearly supports the claim.
   - "FALSE" if at least one document clearly contradicts the claim.
   - "UNVERIFIABLE" if none of the documents provide enough information.

Respond strictly in this JSON format:

{{
  "language": "<language_code>",
  "verdict": "<TRUE | FALSE | UNVERIFIABLE>",
  "final_explanation": "<a brief explanation in the same language as the claim>"
}}

DO NOT guess. If evidence is missing, use "UNVERIFIABLE".
DO NOT provide general knowledge.
No markdown, no introduction. Only output valid JSON.
"""

# Get LLM response
def get_llm_response(prompt):
    response = chat(model="llama3", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

# Fact-Check Button
if st.button("🦖 Fact-Check Claim") and claim:
    with st.spinner("Analyzing claim and retrieving evidence..."):
        docs_for_prompt, titles, fragments_for_display, distances = retrieve_documents(claim)
        prompt = build_prompt(claim, docs_for_prompt)
        raw_llm_response = get_llm_response(prompt)

        try:
            parsed = json.loads(raw_llm_response)
            verdict = parsed["verdict"].upper()
            explanation = parsed["final_explanation"]

            st.markdown("#### 👨‍💼 **User Claim:**")
            st.info(claim)

            if verdict == "TRUE":
                st.success(f"**Verdict: {verdict}**\n\n**Explanation:** {explanation}")
            elif verdict == "FALSE":
                st.error(f"**Verdict: {verdict}**\n\n**Explanation:** {explanation}")
            elif verdict == "UNVERIFIABLE":
                st.warning(f"**Verdict: {verdict}**\n\n**Explanation:** {explanation}")
            else:
                st.info(f"Unknown verdict: {verdict}\n\n{explanation}")

            with st.expander("🔍 Sources"):
                for i, (title, fragment) in enumerate(zip(titles, fragments_for_display)):
                    st.markdown(f"**Document {i+1} - {title}**")
                    st.markdown(fragment)

        except Exception as e:
            st.error("Could not parse model output as valid JSON.")
            st.text("Raw model output:")
            st.code(raw_llm_response)
            st.text(f"Parsing error: {e}")
