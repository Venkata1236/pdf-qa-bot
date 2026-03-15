# streamlit_app.py
# PDF Q&A Bot — Multi-PDF Support
# All PDFs merged into one FAISS index — chat across all at once

import os
import streamlit as st
from dotenv import load_dotenv

from core.loader import load_pdf_from_bytes
from core.splitter import split_documents, get_chunk_stats
from core.embeddings import get_embeddings
from core.vector_store import create_vector_store, merge_vector_stores
from core.retrieval_chain import create_retrieval_chain, ask_question

# ── Load API key ──────────────────────────────────────────────
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="PDF Q&A Bot",
    page_icon="📄",
    layout="centered"
)

# ── Header ────────────────────────────────────────────────────
st.title("📄 PDF Q&A Bot")
st.caption("Upload multiple PDFs and chat across all of them using RAG + FAISS + OpenAI")
st.divider()


# ── Session state init ────────────────────────────────────────
if "chain" not in st.session_state:
    st.session_state.chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "loaded_pdfs" not in st.session_state:
    st.session_state.loaded_pdfs = []     # list of filenames already processed


# ── Helper: Process one PDF into a vector store ───────────────
def process_pdf(file_bytes: bytes, filename: str, embeddings):
    """Loads, splits and embeds a single PDF."""
    documents = load_pdf_from_bytes(file_bytes, filename)
    chunks = split_documents(documents)
    stats = get_chunk_stats(chunks)
    st.write(f"✅ {filename} → {len(documents)} pages → {stats['total_chunks']} chunks")
    vector_store = create_vector_store(chunks, embeddings)
    return vector_store


# ── PDF Input ─────────────────────────────────────────────────
st.subheader("📂 Load PDFs")

input_method = st.radio(
    "How do you want to load PDFs?",
    ["Upload PDFs", "Enter File Paths"],
    horizontal=True
)

file_list = []   # list of (bytes, filename) tuples

# ── Option 1: Upload multiple PDFs ───────────────────────────
if input_method == "Upload PDFs":
    uploaded_files = st.file_uploader(
        "Choose one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload multiple PDFs — they will all be merged into one knowledge base"
    )
    if uploaded_files:
        for f in uploaded_files:
            file_list.append((f.read(), f.name))

# ── Option 2: File Paths ──────────────────────────────────────
elif input_method == "Enter File Paths":
    st.caption("Enter one file path per line")
    paths_input = st.text_area(
        "File paths",
        placeholder="C:\\Users\\bomma\\Downloads\\doc1.pdf\nC:\\Users\\bomma\\Downloads\\doc2.pdf",
        height=120
    )
    if paths_input:
        for path in paths_input.strip().splitlines():
            path = path.strip().strip('"').strip("'")
            if not path:
                continue
            if not os.path.exists(path):
                st.error(f"❌ File not found: {path}")
            elif not path.endswith(".pdf"):
                st.error(f"❌ Not a PDF: {path}")
            else:
                with open(path, "rb") as f:
                    file_list.append((f.read(), os.path.basename(path)))

# ── Build / Update Pipeline ───────────────────────────────────
if file_list:
    new_files = [
        (b, name) for b, name in file_list
        if name not in st.session_state.loaded_pdfs
    ]

    if new_files:
        if not api_key:
            st.error("❌ API key not found. Check your .env file.")
            st.stop()

        with st.status(
            f"⏳ Processing {len(new_files)} new PDF(s)...",
            expanded=True
        ) as status:
            embeddings = get_embeddings(api_key)
            new_vector_stores = []

            for file_bytes, filename in new_files:
                st.write(f"📄 Processing: {filename}")
                vs = process_pdf(file_bytes, filename, embeddings)
                new_vector_stores.append(vs)
                st.session_state.loaded_pdfs.append(filename)

            # Merge with existing vector store if any
            if st.session_state.chain is not None:
                st.write("🔗 Merging with existing knowledge base...")
                existing_retriever = st.session_state.chain.retriever
                existing_vs = existing_retriever.vectorstore
                all_stores = [existing_vs] + new_vector_stores
                merged_vs = merge_vector_stores(all_stores, embeddings)
            else:
                if len(new_vector_stores) > 1:
                    st.write("🔗 Merging all PDFs into one knowledge base...")
                    merged_vs = merge_vector_stores(new_vector_stores, embeddings)
                else:
                    merged_vs = new_vector_stores[0]

            st.write("🔗 Building retrieval chain...")
            st.session_state.chain = create_retrieval_chain(merged_vs, api_key)

            status.update(
                label=f"✅ {len(st.session_state.loaded_pdfs)} PDF(s) ready!",
                state="complete"
            )

    # Show loaded PDFs
    st.success(
        f"📚 Knowledge base: {len(st.session_state.loaded_pdfs)} PDF(s) loaded — "
        + ", ".join(st.session_state.loaded_pdfs)
    )

    # Reset button
    if st.button("🗑️ Clear all PDFs and start over"):
        st.session_state.chain = None
        st.session_state.chat_history = []
        st.session_state.loaded_pdfs = []
        st.rerun()

# ── Chat Interface ────────────────────────────────────────────
if st.session_state.chain:
    st.divider()
    st.subheader("💬 Ask questions across all PDFs")

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg:
                with st.expander("📚 Source pages used"):
                    for src in msg["sources"]:
                        page = src.metadata.get("page", 0)
                        source = src.metadata.get("source", "unknown")
                        filename = os.path.basename(source)
                        st.write(f"📄 {filename} — Page {page + 1}")
                        st.caption(src.page_content[:200] + "...")

    # Chat input
    question = st.chat_input("Ask anything across all your PDFs...")

    if question:
        with st.chat_message("user"):
            st.write(question)
        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    result = ask_question(st.session_state.chain, question)
                    answer = result["answer"]
                    sources = result["source_chunks"]

                    st.write(answer)

                    with st.expander("📚 Source pages used"):
                        for src in sources:
                            page = src.metadata.get("page", 0)
                            source = src.metadata.get("source", "unknown")
                            filename = os.path.basename(source)
                            st.write(f"📄 {filename} — Page {page + 1}")
                            st.caption(src.page_content[:200] + "...")

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })

                except Exception as e:
                    st.error(f"❌ Error: {e}")

else:
    st.info("👆 Upload PDFs or enter file paths to get started")