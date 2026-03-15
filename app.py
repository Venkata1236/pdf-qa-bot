# app.py
# Main entry point — PDF Q&A Bot
# Concepts: Document Loaders, Text Splitters, FAISS, Retrieval Chain

import os
from dotenv import load_dotenv

from core.loader import load_pdf_from_path
from core.splitter import split_documents, get_chunk_stats
from core.embeddings import get_embeddings
from core.vector_store import create_vector_store
from core.retrieval_chain import create_retrieval_chain, ask_question

# ── Load API key ──────────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    print("❌ ERROR: Please set your OPENAI_API_KEY in the .env file.")
    exit(1)


def build_qa_pipeline(pdf_path: str):
    """
    Builds the full RAG pipeline from a PDF file.

    Steps:
    1. Load PDF → Documents
    2. Split → Chunks
    3. Embed → Vectors
    4. Store → FAISS
    5. Build → RetrievalQA chain

    Args:
        pdf_path: path to the PDF file

    Returns:
        RetrievalQA chain ready to answer questions
    """
    print("\n📄 Loading PDF...")
    documents = load_pdf_from_path(pdf_path)

    print("\n✂️  Splitting into chunks...")
    chunks = split_documents(documents)
    stats = get_chunk_stats(chunks)
    print(f"   Stats: {stats}")

    print("\n🔢 Creating embeddings...")
    embeddings = get_embeddings(API_KEY)

    print("\n🗄️  Building FAISS vector store...")
    vector_store = create_vector_store(chunks, embeddings)

    print("\n🔗 Building retrieval chain...")
    chain = create_retrieval_chain(vector_store, API_KEY)

    print("\n✅ Pipeline ready! Ask your questions.\n")
    return chain


def main():
    print("\n" + "=" * 60)
    print("         📄 PDF Q&A BOT")
    print("    LangChain + FAISS + OpenAI RAG")
    print("=" * 60)

    # ── Get PDF path ──────────────────────────────────────────
    pdf_path = input("\n📂 Enter path to your PDF file: ").strip()

    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        exit(1)

    if not pdf_path.endswith(".pdf"):
        print("❌ Please provide a PDF file.")
        exit(1)

    # ── Build pipeline ────────────────────────────────────────
    chain = build_qa_pipeline(pdf_path)

    print("-" * 60)
    print("💬 Ask questions about your PDF!")
    print("   Type 'quit' to exit")
    print("-" * 60 + "\n")

    # ── Chat loop ─────────────────────────────────────────────
    while True:
        try:
            question = input("You: ").strip()

            if not question:
                continue

            if question.lower() == "quit":
                print("\n👋 Bye!")
                break

            print("\n🤔 Thinking...\n")
            result = ask_question(chain, question)

            print(f"🤖 Answer:\n{result['answer']}")

            # Show source chunks used
            print(f"\n📚 Sources used: {len(result['source_chunks'])} chunks")
            for i, chunk in enumerate(result['source_chunks'], 1):
                page = chunk.metadata.get('page', 'unknown')
                print(f"   Chunk {i} → Page {page + 1}")

            print()

        except KeyboardInterrupt:
            print("\n\n👋 Bye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    main()