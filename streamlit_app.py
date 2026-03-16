# streamlit_app.py
# PDF Q&A Bot — Multi-PDF Support

import os
import streamlit as st
from dotenv import load_dotenv

from core.loader import load_pdf_from_bytes
from core.splitter import split_documents, get_chunk_stats
from core.embeddings import get_embeddings
from core.vector_store import create_vector_store, merge_vector_stores
from core.retrieval_chain import create_retrieval_chain, ask_question

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")

st.set_page_config(page_title="PDF Q&A Bot", page_icon="📄", layout="centered")
st.title("📄 PDF Q&A Bot")
st.caption("Upload multiple PDFs and chat across all of them using RAG + FAISS + OpenAI")
st.divider()

if "chain_tuple" not in st.session_state:
    st.session_state.chain_tuple = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "loaded_pdfs" not in st.session_state:
    st.session_state.loaded_pdfs = []
if "all_vector_stores" not in st.session_state:
    st.session_state.all_vector_stores = []


def process_pdf(file_bytes, filename, embeddings):
    documents = load_pdf_from_bytes(file_bytes, filename)
    chunks = split_documents(documents)
    stats = get_chunk_stats(chunks)
    vector_store = create_vector_store(chunks, embeddings)
    return vector_store, len(documents), stats['total_chunks']


st.subheader("📂 Load PDFs")
input_method = st.radio("How do you want to load PDFs?", ["Upload PDFs", "Enter File Paths"], horizontal=True)

file_list = []

if input_method == "Upload PDFs":
    uploaded_files = st.file_uploader("Choose one or more PDF files", type=["pdf"], accept_multiple_files=True)
    if uploaded_files:
        for f in uploaded_files:
            file_list.append((f.read(), f.name))

elif input_method == "Enter File Paths":
    st.caption("Enter one file path per line")
    paths_input = st.text_area("File paths", placeholder="C:\\Users\\bomma\\Downloads\\doc1.pdf", height=120)
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

if file_list:
    new_files = [(b, name) for b, name in file_list if name not in st.session_state.loaded_pdfs]

    if new_files:
        if not api_key:
            st.error("❌ API key not found. Check your .env file.")
            st.stop()

        progress = st.progress(0, text="⏳ Starting...")
        status_box = st.empty()

        try:
            status_box.info("🔢 Loading embeddings model...")
            embeddings = get_embeddings(api_key)
            progress.progress(10, text="🔢 Embeddings model ready")

            total = len(new_files)
            for i, (file_bytes, filename) in enumerate(new_files):
                status_box.info(f"📄 Processing: {filename}")
                vs, pages, chunks = process_pdf(file_bytes, filename, embeddings)
                st.session_state.all_vector_stores.append(vs)
                st.session_state.loaded_pdfs.append(filename)
                pct = 10 + int(((i + 1) / total) * 70)
                progress.progress(pct, text=f"✅ {filename} → {pages} pages → {chunks} chunks")

            status_box.info("🔗 Merging all PDFs into one knowledge base...")
            progress.progress(85, text="🔗 Merging knowledge base...")
            all_stores = st.session_state.all_vector_stores
            merged_vs = merge_vector_stores(all_stores, embeddings) if len(all_stores) > 1 else all_stores[0]

            status_box.info("🔗 Building retrieval chain...")
            progress.progress(95, text="🔗 Building retrieval chain...")
            st.session_state.chain_tuple = create_retrieval_chain(merged_vs, api_key)

            progress.progress(100, text="✅ Pipeline ready!")
            status_box.success(f"✅ Done! {len(st.session_state.loaded_pdfs)} PDF(s) loaded and ready.")

        except Exception as e:
            status_box.error(f"❌ Error: {e}")

    if st.session_state.loaded_pdfs:
        st.success(f"📚 {len(st.session_state.loaded_pdfs)} PDF(s) loaded — " + ", ".join(st.session_state.loaded_pdfs))

    if st.button("🗑️ Clear all PDFs and start over"):
        st.session_state.chain_tuple = None
        st.session_state.chat_history = []
        st.session_state.loaded_pdfs = []
        st.session_state.all_vector_stores = []
        st.rerun()

if st.session_state.chain_tuple:
    st.divider()
    st.subheader("💬 Ask questions across all PDFs")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg:
                with st.expander("📚 Source pages used"):
                    for src in msg["sources"]:
                        page = src.metadata.get("page", 0)
                        source = src.metadata.get("source", "unknown")
                        fname = os.path.basename(source)
                        st.write(f"📄 {fname} — Page {page + 1}")
                        st.caption(src.page_content[:200] + "...")

    question = st.chat_input("Ask anything across all your PDFs...")

    if question:
        with st.chat_message("user"):
            st.write(question)
        st.session_state.chat_history.append({"role": "user", "content": question})

        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    result = ask_question(st.session_state.chain_tuple, question)
                    answer = result["answer"]
                    sources = result["source_chunks"]
                    st.write(answer)
                    with st.expander("📚 Source pages used"):
                        for src in sources:
                            page = src.metadata.get("page", 0)
                            source = src.metadata.get("source", "unknown")
                            fname = os.path.basename(source)
                            st.write(f"📄 {fname} — Page {page + 1}")
                            st.caption(src.page_content[:200] + "...")
                    st.session_state.chat_history.append({"role": "assistant", "content": answer, "sources": sources})
                except Exception as e:
                    st.error(f"❌ Error: {e}")

else:
    st.info("👆 Upload PDFs or enter file paths to get started")