# core/embeddings.py
# Converts text chunks into vector embeddings
# Concept: Embeddings — turning text into numbers

from langchain_openai import OpenAIEmbeddings


def get_embeddings(api_key: str) -> OpenAIEmbeddings:
    """
    Creates and returns an OpenAI embeddings model.

    What embeddings do:
    - Convert text chunks into vectors (lists of numbers)
    - Similar text = similar vectors
    - This is how semantic search works

    Example:
    "Python programming" → [0.23, -0.51, 0.87, ...]  (1536 numbers)
    "Python coding"      → [0.24, -0.49, 0.85, ...]  (very similar!)
    "I love pizza"       → [0.91,  0.32, -0.44, ...]  (very different)

    Args:
        api_key: OpenAI API key

    Returns:
        OpenAIEmbeddings model
    """
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",  # OpenAI's embedding model
        openai_api_key=api_key
    )

    return embeddings

