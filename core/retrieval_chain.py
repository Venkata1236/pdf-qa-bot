# core/retrieval_chain.py
# Retrieves relevant chunks and generates answer using OpenAI
# Concept: Retrieval Chain — the final step of RAG

from langchain_openai import ChatOpenAI
from langchain_community.chains import RetrievalQAWithSourcesChain
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS


def create_retrieval_chain(
    vector_store: FAISS,
    api_key: str,
    temperature: float = 0.3,
    k: int = 4
) -> RetrievalQA:
    """
    Creates a RetrievalQA chain that:
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
        RetrievalQA chain ready to answer questions
    """

    # Custom prompt — tells the model to use only the PDF context
    prompt_template = """
You are a helpful assistant that answers questions based on the provided PDF document.

Use ONLY the following context to answer the question.
If the answer is not in the context, say "I couldn't find that information in the PDF."
Do not make up answers or use outside knowledge.

Context:
{context}

Question:
{question}

Answer:
"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # LLM for generating answers
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=temperature,
        openai_api_key=api_key,
        max_tokens=800
    )

    # Retriever — searches FAISS for relevant chunks
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

    # RetrievalQA — connects retriever + LLM + prompt
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",         # "stuff" = stuff all chunks into one prompt
        retriever=retriever,
        return_source_documents=True,  # Returns which chunks were used
        chain_type_kwargs={"prompt": prompt}
    )

    return chain


def ask_question(chain: RetrievalQA, question: str) -> dict:
    """
    Asks a question using the retrieval chain.

    Args:
        chain   : RetrievalQA chain
        question: user's question

    Returns:
        dict with 'answer' and 'source_chunks'
    """
    result = chain.invoke({"query": question})

    return {
        "answer": result["result"],
        "source_chunks": result["source_documents"]
    }