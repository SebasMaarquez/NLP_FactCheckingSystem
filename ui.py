import streamlit as st
import chromadb
from ollama import chat

# === Setup page ===
st.set_page_config(page_title="Scientific Fact Checker", page_icon="🧠")
st.title("🧠 Scientific Fact Checker")
st.write("Enter a scientific claim and get a verdict based on PubMed evidence")

# === Claim Input ===
claim = st.text_input("🔍 Enter a claim to fact-check:", placeholder="e.g. Alpha-bisabolol inhibits pepsin.")

# === ChromaDB Connection ===
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection("pubmed_articles_mxbai")

# === Utilities ===
def retrieve_documents(claim, k=3):
    results = collection.query(query_texts=[claim], n_results=k)
    return results["documents"][0]

def get_llm_verdict(claim, docs):
    context = "\n\n".join(docs)
    prompt = f"""
    Given the scientific claim: "{claim}"

    And the following evidence from biomedical literature:
    {context}

    Decide if the claim is TRUE, FALSE, or UNVERIFIABLE.
    Provide a brief explanation, using ONLY the evidence above.
    """
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
        llm_response = get_llm_verdict(claim, docs)
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
