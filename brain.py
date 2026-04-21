import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

def get_llm():
    """Main writer/reasoner — Llama 3.3 70B via GitHub Models (150/day, own quota)."""
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise ValueError("GITHUB_TOKEN not found!")
    return ChatOpenAI(
        model="Llama-3.3-70B-Instruct",
        api_key=token,
        base_url="https://models.inference.ai.azure.com",
        temperature=0.1,
        max_tokens=4096
    )

def get_mini_llm():
    """Fast workhorse for extraction, query rewriting, routing — gpt-4o-mini (150/day, separate quota)."""
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise ValueError("GITHUB_TOKEN not found!")
    return ChatOpenAI(
        model="gpt-4o-mini",
        api_key=token,
        base_url="https://models.inference.ai.azure.com",
        temperature=0.1,
        max_tokens=1024
    )

def get_embeddings():
    # Embeddings use local SentenceTransformer in tools.py — this function is unused
    pass
