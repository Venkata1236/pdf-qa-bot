# 📄 PDF Q&A Bot

> Upload multiple PDFs and chat across all of them using RAG + FAISS + OpenAI

![Python](https://img.shields.io/badge/Python-3.11-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.3.7-green)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--3.5--turbo-orange)
![FAISS](https://img.shields.io/badge/FAISS-1.8.0-purple)
![Streamlit](https://img.shields.io/badge/Streamlit-1.41.1-red)

---

## 📌 What Is This?

Upload one or multiple PDF files and ask questions across all of them. Built on a full RAG (Retrieval-Augmented Generation) pipeline — documents are chunked, embedded, stored in FAISS, and retrieved semantically to answer questions accurately.

---

## 🗺️ Simple Flow

```
Upload PDF(s)
      ↓
loader.py       → read PDF text page by page
      ↓
splitter.py     → split into overlapping chunks
      ↓
embeddings.py   → convert chunks to vectors
      ↓
vector_store.py → store vectors in FAISS index
      ↓
You ask a question
      ↓
FAISS finds most relevant chunks
      ↓
retrieval_chain → chunks + question → OpenAI
      ↓
Answer + source pages shown
```

---

## 🏗️ Detailed Architecture

```
User
 ├── streamlit_app.py   → Web UI (upload PDF or enter file path)
 └── app.py             → Terminal interface (enter file path)
          │
          ▼
      core/
      ├── loader.py          → PyPDFLoader → List of Documents
      ├── splitter.py        → RecursiveCharacterTextSplitter → Chunks
      ├── embeddings.py      → OpenAIEmbeddings (ada-002) → Vectors
      ├── vector_store.py    → FAISS → store, search, merge, save, load
      └── retrieval_chain.py → RetrievalQA → retrieve + generate answer
          │
          ▼
      OpenAI API
      gpt-3.5-turbo → generates answer from retrieved chunks

.env              → API key
requirements.txt  → all libraries
```

---

## 📁 Project Structure

```
pdf_qa_bot/
├── app.py                   ← Terminal version
├── streamlit_app.py         ← Web UI (deploy this)
├── core/
│   ├── __init__.py
│   ├── loader.py            ← Load PDF → Documents
│   ├── splitter.py          ← Documents → Chunks
│   ├── embeddings.py        ← Text → Vectors
│   ├── vector_store.py      ← Store + Search FAISS
│   └── retrieval_chain.py   ← Retrieve + Answer
├── .env                     ← API key (never push!)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🧠 Key Concepts

| Concept | What It Does |
|---|---|
| **Document Loader** | Reads PDF page by page into LangChain Document objects |
| **Text Splitter** | Splits text into overlapping chunks for precise retrieval |
| **Chunk Overlap** | Shared characters between chunks so context isn't lost at boundaries |
| **Embeddings** | Converts text to 1536-dimension vectors using OpenAI ada-002 |
| **FAISS** | Facebook's vector database — stores and searches embeddings by similarity |
| **RetrievalQA** | LangChain chain that retrieves relevant chunks and generates answer |
| **Multi-PDF Merge** | Multiple FAISS indexes merged into one for cross-document search |

---

## 📊 Chunking Strategy

```
chunk_size    = 1000 chars  ← max size per chunk
chunk_overlap = 200 chars   ← shared between adjacent chunks

Example:
Chunk 1: chars 0    → 1000
Chunk 2: chars 800  → 1800  ← 200 overlap with chunk 1
Chunk 3: chars 1600 → 2600  ← 200 overlap with chunk 2
```

Split priority: paragraph → line → sentence → word

---

## 🌡️ Temperature

| Chain | Temperature | Reason |
|---|---|---|
| retrieval_chain | 0.3 | Factual, grounded answers from PDF only |

---

## ⚙️ Local Setup

**Step 1 — Clone the repo:**
```bash
git clone https://github.com/YOUR_USERNAME/pdf-qa-bot.git
cd pdf_qa_bot
```

**Step 2 — Install dependencies:**
```bash
pip install -r requirements.txt
```

**Step 3 — Add your OpenAI API key in `.env`:**
```
OPENAI_API_KEY=sk-your-actual-key-here
```

**Step 4 — Run:**

Streamlit UI:
```bash
python -m streamlit run streamlit_app.py
```

Terminal version:
```bash
python app.py
```

---

## 🚀 Deploy on Streamlit Cloud

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → select `streamlit_app.py`
4. Go to Settings → Secrets → add:
```toml
OPENAI_API_KEY = "sk-your-key-here"
```
5. Click Deploy ✅

---

## 📦 Tech Stack

- **LangChain** — Document loaders, text splitters, RetrievalQA
- **OpenAI** — GPT-3.5-turbo + text-embedding-ada-002
- **FAISS** — Vector store for semantic search
- **pypdf** — PDF text extraction
- **Streamlit** — Web UI
- **python-dotenv** — API key management

---

## 👤 Author

**Venkata Reddy Bommavaram**
- 📧 bommavaramvenkat2003@gmail.com
- 💼 [LinkedIn](https://linkedin.com/in/venkatareddy1203)
- 🐙 [GitHub](https://github.com/venkata1236)