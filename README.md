# NLP---FactCheckingSystem
The repository of the final NLP project of Sebastián Márquez and David Gómez

# Scientific Fact Checker (Biomedical RAG System)

This project implements a **scientific fact-checking system** that uses a Retrieval-Augmented Generation (RAG) pipeline to verify biomedical claims using evidence from 3500 **PubMed abstracts**. It integrates **embeddings**, a **vector database (ChromaDB)**, and a **local LLM (LLaMA 3 via Ollama)**, presented through a **Streamlit UI**.

---

## How It Works

1. **User inputs a scientific claim** in natural language.
2. The claim is embedded and used to retrieve top-k relevant PubMed abstracts from ChromaDB.
3. A carefully constructed prompt is sent to the LLM with the claim and evidence.
4. The LLM returns a verdict: **TRUE**, **FALSE**, or **UNVERIFIABLE**, with a short explanation.
5. Retrieved evidence is displayed, including document titles and the most relevant sentence fragments.

---

## Scripts Overview

| Script | Description |
|--------|-------------|
| `parse_pubmed.py` | Parses the original compressed PubMed XML files and extracts titles and abstracts into a usable format. |
| `index_pubmed_articles.py` | Embeds the first 3500 articles using `mxbai-embed-large-v1` and stores them in a persistent ChromaDB collection. |
| `ui.py` | The main Streamlit interface. Lets users input a claim, retrieves evidence, builds prompts, and displays LLM verdicts and sources. |
| `recall_at_k.py` | Evaluates the document retrieval system using **Recall@K**. Compares retrieved documents with ground-truth reference snippets. |
| `evaluation_set.json` | A small handcrafted evaluation dataset of claims with corresponding relevant sentences from PubMed. Used for evaluation metrics. |
| `build_eval_set.py` | This script facilitates the construction of the evaluation dataset used to compute Recall@K. |

---

## ⚙️ Technologies Used

- **Embeddings:** [`mixedbread-ai/mxbai-embed-large-v1`](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1)
- **Vector DB:** [ChromaDB](https://www.trychroma.com/)
- **LLM:** [LLaMA 3](https://ollama.com/library/llama3) via [Ollama](https://ollama.com/)
- **UI Framework:** [Streamlit](https://streamlit.io/)
- **Sentence Embeddings Comparison:** `scikit-learn` (cosine similarity)
- **Language Detection:** Prompting Engineering

---

## Evaluation Summary

| Metric | Result |
|--------|--------|
| Recall@1 | 0.40 |
| Recall@3 | 0.70 |
| Recall@5 | 1.00 |
| Recall@10 | 1.00 |

---
