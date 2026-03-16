# core/retrieval_chain.py
# Retrieves relevant chunks and generates answer using OpenAI
# Uses modern LCEL (LangChain Expression Language) — no deprecated chains

import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS


def create_retrieval_chain(
    vector_store: FAISS,
    api_key: str,
    temperature: float = 0.3,
    k: int = 4
):
    """
    Creates a retrieval chain using modern LCEL syntax.

    Steps:
    1. Converts question to vector
    2. Finds k most relevant chunks from FAISS
    3. Sends chunks + question to OpenAI
    4. Returns the answer

    Args:
        vector_store: FAISS vector store with embedded chunks
        api_key     : OpenAI API key
        temperature : response creativity (default 0.3)
        k           : number of chunks to retrieve (default 4)

    Returns:
        LCEL chain ready to answer questions
    """

    # Custom prompt — answers only from PDF context
    prompt_template = PromptTemplate.from_template("""
You are a helpful assistant that answers questions based on the provided PDF document.

Use ONLY the following context to answer the question.
If the answer is not in the context, say "I couldn't find that information in the PDF."
Do not make up answers or use outside knowledge.

Context:
{context}

Question:
{question}

Answer:
""")

    # LLM
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=temperature,
        openai_api_key=api_key,
        max_tokens=800
    )

    # Retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    # LCEL chain: retrieve → format → prompt → llm → parse
    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt_template
        | llm
        | StrOutputParser()
    )

    # Store retriever on chain for source document access
    return chain, retriever


def ask_question(chain_tuple, question: str) -> dict:
    chain, retriever = chain_tuple
    answer = chain.invoke(question)
    source_chunks = retriever.invoke(question)
    return {
        "answer": answer,
        "source_chunks": source_chunks
    }