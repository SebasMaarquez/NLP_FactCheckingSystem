import streamlit as st
import chromadb
from ollama import chat
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from nltk.tokenize import sent_tokenize
import time
import json
import pandas as pd

# Streamlit page setup
st.set_page_config(page_title="Scientific Fact Checker", page_icon="🧠")
st.markdown("""
    <style>
        .stApp {
            background-color: #121212 !important;
            color: #ffffff;
        }
        [data-testid="stHeader"] {
            background-color: #1e1e1e !important;
        }
        .block-container {
            padding: 2rem;
        }
        .stButton>button {
            color: white;
            background-color: #333333;
            border-radius: 8px;
            padding: 0.5em 1em;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #555555;
        }
        .stMetric {
            background-color: #2c2c2c;
            color: #ffffff;
            padding: 1em;
            border-radius: 10px;
            box-shadow: 0px 0px 5px #000000;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🧠 MyHealth")
st.markdown("""
#### 🤖 What does this do?
Welcome to MyHealth: Your Scientific Fact Checker! Powered by AI and real scientific literature from PubMed, this tool helps you verify health-related claims fast. Just type a scientific statement (like *"Vitamin D prevents the flu"*) and our chatbot will tell you if it's **true**, **false**, or **unverifiable**.
""")

# Claim input
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

# Get embeddings
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings[0].numpy().tolist()

# ChromaDB connection
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection("pubmed_articles_mxbai")

def retrieve_documents(claim, k=3):
    embedding = get_embedding(claim)
    results = collection.query(query_embeddings=[embedding], n_results=k, include=["documents", "distances"])
    docs = results["documents"][0]
    distances = results["distances"][0]

    titles, best_fragments = [], []
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

def get_llm_verdict(prompt):
    response = chat(model="llama3", messages=[{"role": "user", "content": prompt}], options={"temperature": 0.3})
    return response["message"]["content"]

def build_summary_prompt(best_fragments):
    return f"""
You are a scientific summarization assistant.

Summarize the following scientific text fragments in a concise paragraph.

Do NOT interpret or judge any claim. Just summarize what the documents say.

Text to summarize:
{chr(10).join(best_fragments)}
"""

def get_summary(prompt):
    response = chat(model="llama3", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"].strip()

# Main logic
if st.button("🦖 Fact-Check Claim") and claim:
    start_time = time.time()
    with st.spinner("Analyzing claim and retrieving evidence..."):
        docs, titles, best_fragments, distances = retrieve_documents(claim)
        prompt = build_prompt(claim, docs)
        llm_response = get_llm_verdict(prompt)
        summary_prompt = build_summary_prompt(best_fragments)
        summary = get_summary(summary_prompt)

        try:
            parsed = json.loads(llm_response)
            verdict = parsed["verdict"].upper()
            explanation = parsed["final_explanation"]

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

            if verdict != "UNVERIFIABLE":
                with st.expander("🔍 Sources"):
                    for i, (title, fragment) in enumerate(zip(titles, best_fragments)):
                        st.markdown(f"**Document {i+1} - {title}**")
                        st.markdown(fragment)

                st.markdown("#### 🧾 Summary of Retrieved Evidence")
                st.info(summary)


            # NLI model comparison
            combined_evidence = " ".join(best_fragments)
            nli_verdict, nli_probs = run_nli_verdict(claim, combined_evidence)

            comparison_df = pd.DataFrame([
                {"System": "MyHealth (LLaMA3)", "Verdict": verdict, "Explanation": explanation},
                {"System": "External NLI Model", "Verdict": nli_verdict, "Explanation": f"Probabilities: {nli_probs}"}
            ])

            st.markdown("### 📊 Comparison Between Systems")
            st.table(comparison_df)

        except Exception as e:
            st.error("Could not parse model output as valid JSON.")
            st.text("Raw model output:")
            st.code(llm_response)
            st.text(f"Parsing error: {e}")

    elapsed_seconds = round(time.time() - start_time, 2)
    minutes, seconds = divmod(elapsed_seconds, 60)
    st.info(f"⏱️ Time taken: {int(minutes)} minutes and {int(seconds)} seconds")
