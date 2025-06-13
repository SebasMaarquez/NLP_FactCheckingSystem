#GitHub token to commit from terminal > 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
import streamlit as st
import chromadb
from ollama import chat
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import os
from sklearn.metrics.pairwise import cosine_similarity
import re
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize 
import time
import json
import pandas as pd

nltk.download('punkt_tab')

os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

# Setup page 
st.set_page_config(page_title="Scientific Fact Checker", page_icon="🧠")
st.markdown("""
    <style>
        /* Set main container background */
        .stApp {
            background-color: #e6f2ff !important;
        }
        /* Set header background */
        [data-testid="stHeader"] {
            background-color: #e6f2ff !important;
        }
        .block-container {
            padding: 2rem;
        }
        .stButton>button {
            color: white;
            background-color: #007acc;
            border-radius: 8px;
            padding: 0.5em 1em;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #005f99;
        }
        .stMetric {
            background-color: #f0f8ff;
            padding: 1em;
            border-radius: 10px;
            box-shadow: 0px 0px 5px lightgray;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🧠 MyHealth")
st.markdown("""
#### 🤖 What does this do?

Welcome to MyHealth: Your Scientific Fact Checker! Powered by AI and real scientific literature from PubMed, this tool helps you verify health-related claims — fast, smart, and with style. Just type a scientific statement (like *"Vitamin D prevents the flu"*) and our chatbot will scan the research, weigh the evidence, and tell you if it's **true**, **false**, or **unverifiable** — all in language you can understand.

Try it out and explore the science behind everyday health!
""")
st.write("Enter a scientific claim and get a verdict based on PubMed evidence")

# Claim Input 
claim = st.text_input("🔍 Enter a claim to fact-check:", placeholder="e.g. Alpha-bisabolol inhibits pepsin.")

# Load embedding model 
model_name = "mixedbread-ai/mxbai-embed-large-v1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Load NLI model (public, general-purpose NLI + FEVER model)
@st.cache_resource
def load_nli_model():
    model_path = "ynie/bart-large-snli_mnli_fever_anli_R1_R2_R3-nli"
    tokenizer_nli = AutoTokenizer.from_pretrained(model_path)
    model_nli = AutoModelForSequenceClassification.from_pretrained(model_path)
    return tokenizer_nli, model_nli

def get_external_nli_model():
    if "external_nli_model" not in st.session_state:
       st.session_state["external_nli_model"] = load_nli_model()
        
    return st.session_state["external_nli_model"]


tokenizer_nli, model_nli = load_nli_model()
id2label_nli = {0: "FALSE", 1: "UNVERIFIABLE", 2: "TRUE"}

def run_nli_verdict(claim, evidence):
    input_text = f"{claim} </s> {evidence}"
    inputs = tokenizer_nli(input_text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model_nli(**inputs).logits
    probs = F.softmax(logits, dim=1).squeeze().numpy()
    prediction = id2label_nli[int(np.argmax(probs))]
    return prediction, {k: float(v) for k, v in zip(id2label_nli.values(), probs.round(3))}


def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings[0].numpy().tolist()

# Connect to ChromaDB 
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection("pubmed_articles_mxbai")

# Retrieve Documents 
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

# LLM Prompt Construction 
def build_prompt(claim, docs):

    doc_blocks = []
    for i, doc in enumerate(docs):
        if ". " in doc:
            title, abstract = doc.split(". ", 1)
        else:
            title, abstract = "Untitled", doc

        doc_blocks.append(f"Document {i+1} (Title: {title.strip()}):\n{abstract.strip()}")

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

# Get Verdict from the LLM 
def get_llm_verdict(prompt):
    response = chat(model="llama3:8b", messages=[{"role": "user", "content": prompt}], options={"temperature":0.5})
    return response["message"]["content"]


def build_summary_prompt(claim, best_fragments):
    return f"""
You are a scientific summarization assistant.

Your ONLY task is to return a single paragraph summary that distills factual evidence from the provided scientific documents. Do not verify, confirm, or argue the claim’s truth; simply summarize the most relevant factual evidence found within the retrieved documents.

 IMPORTANT INSTRUCTIONS:
- Detect the language of the claim and provide your response in that language.
- DO NOT explain what you're doing.
- DO NOT include any introductions, notes, or headers.
- DO NOT say anything before or after the paragraph.
- Just return the paragraph, nothing more.
- Do not add phrases like "Here is the summary", "Summary:", etc.

 TASK:
From the evidence provided below, write a concise summary that highlights the most relevant factual information related to the claim.
- The paragraph must have no more than 5 sentences.
- Use only the information provided below from the retrieved documents.
- Do NOT verify or discuss if the claim is true or false.

Claim:
"{claim}"

Scientific Documents:
{chr(10).join(best_fragments)}
"""

def get_summary(prompt):
    response = chat(model="llama3:8b", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"].strip()

def run_external_nli_with_explanation(claim, best_fragments):
    combined_evidence = " ".join(best_fragments)
    # Load the external NLI model (this uses the deferred loading function)
    tokenizer_nli, model_nli = get_external_nli_model()
    
    input_text = f"{claim} </s> {combined_evidence}"
    inputs = tokenizer_nli(input_text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model_nli(**inputs).logits
    probs = F.softmax(logits, dim=1).squeeze().numpy()
    prediction = id2label_nli[int(np.argmax(probs))]
    verdict = prediction.upper()
    
    # Build an explanation string that includes probabilities.
    explanation = (
         f"External NLI analysis indicates the claim is likely {verdict}. "
         f"Probabilities: TRUE={probs[2]:.3f}, FALSE={probs[0]:.3f}, UNVERIFIABLE={probs[1]:.3f}. "
         "This conclusion is drawn from the aggregated scientific evidence."
    )
    
    # Create a naïve summary: take the first sentence of each evidence fragment.
    summary_sentences = []
    for fragment in best_fragments:
        sents = sent_tokenize(fragment)
        if sents:
            summary_sentences.append(sents[0])
    summary = " ".join(summary_sentences)
    return verdict, explanation, summary


# Add a radio button to allow the user to choose the system.
system_choice = st.radio("Choose Fact-Checking System", options=["MyHealth", "External NLI model"])

if st.button("🦖 Fact-Check Claim") and claim:
    start_time = time.time()
    
    # Retrieve documents and evidence fragments.
    with st.spinner("Retrieving evidence..."):
        docs, titles, best_fragments, distances = retrieve_documents(claim)
    
    if system_choice == "MyHealth":
        # Build the prompt and call the chat-based LLM (LLaMA3) approach.
        prompt = build_prompt(claim, docs)
        llm_response = get_llm_verdict(prompt)
        summary_prompt = build_summary_prompt(claim, docs)
        summary = get_summary(summary_prompt)
        
        try:
            parsed = json.loads(llm_response)
            verdict = parsed["verdict"].upper()
            explanation = parsed["final_explanation"]
            
            # Display results for MyHealth system.
            st.markdown("#### 👨‍💼 **User Claim:**")
            st.info(claim)
            st.markdown("### ✅ Fact-Check Result")
            if verdict == "TRUE":
                st.success(f"**Verdict: {verdict}**\n\n**Explanation:** {explanation}")
            elif verdict == "FALSE":
                st.error(f"**Verdict: {verdict}**\n\n**Explanation:** {explanation}")
            elif verdict == "UNVERIFIABLE":
                st.warning(f"**Verdict: {verdict}**\n\n**Explanation:** {explanation}")
            else:
                st.info(f"Unknown verdict: {verdict}\n\n{explanation}")
            
            # Display source documents.
            with st.expander("🔍 Sources"):
                for i, (title, fragment) in enumerate(zip(titles, best_fragments)):
                    st.markdown(f"**Document {i+1} - {title}**")
                    st.markdown(fragment)
            
            st.markdown("#### 🧾 Summary of Retrieved Evidence")
            st.info(summary)
        except Exception as e:
            st.error("Could not parse model output as valid JSON.")
            st.text("Raw model output:")
            st.code(llm_response)
            st.text(f"Parsing error: {e}")
    
    elif system_choice == "External NLI model":
        # Call the external NLI branch to get verdict, explanation, and summary.
        verdict, ext_explanation, ext_summary = run_external_nli_with_explanation(claim, best_fragments)
        st.markdown("#### 👨‍💼 **User Claim:**")
        st.info(claim)
        st.markdown("### ✅ External NLI Model Result")
        if verdict == "TRUE":
            st.success(f"**Verdict: {verdict}**\n\n**Explanation:** {ext_explanation}")
        elif verdict == "FALSE":
            st.error(f"**Verdict: {verdict}**\n\n**Explanation:** {ext_explanation}")
        elif verdict == "UNVERIFIABLE":
            st.warning(f"**Verdict: {verdict}**\n\n**Explanation:** {ext_explanation}")
        else:
            st.info(f"Unknown verdict: {verdict}\n\n{ext_explanation}")
        
        st.markdown("#### 🧾 Summary of Retrieved Evidence")
        st.info(ext_summary)
        with st.expander("🔍 Sources"):
            for i, (title, fragment) in enumerate(zip(titles, best_fragments)):
                st.markdown(f"**Document {i+1} - {title}**")
                st.markdown(fragment)
    
    end_time = time.time()
    elapsed_seconds = round(end_time - start_time, 2)
    minutes, seconds = divmod(elapsed_seconds, 60)
    st.info(f"⏱️ Time taken: {int(minutes)} minutes and {int(seconds)} seconds")
