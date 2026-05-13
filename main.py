from __future__ import annotations

import os
from dotenv import load_dotenv
import openai
from config.config import load_config
from qdrant import QdrantRAG
from tavily import TavilyWebSearch

load_dotenv()

# Load and validate config
config = load_config()
PDF_DATA_DIR = config.qdrant.pdf_data_dir
QDRANT_LOCAL_PATH = config.qdrant.local_path
TOP_K = config.qdrant.top_k
WEB_TOP_K = config.generation.web_top_k
MAX_OUTPUT_TOKENS = config.generation.max_output_tokens
MAX_WEB_CONTEXT_CHARS = config.generation.max_web_context_chars
MAX_RAG_CONTEXT_CHARS = config.generation.max_rag_context_chars
OPENAI_MODEL = config.generation.openai_model


def openai_infer(prompt: str, model: str, api_key: str, max_tokens: int, stream: bool = False) -> str:
    client = openai.OpenAI(api_key=api_key)
    if stream:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            stream=True,
        )
        full = ""
        for chunk in response:
            delta = chunk.choices[0].delta.content if chunk.choices[0].delta else None
            if delta:
                print(delta, end="", flush=True)
                full += delta
        print()  # Newline after streaming
        return full
    else:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""


if __name__ == "__main__":
    user_prompt = "Explain retrieval-augmented generation and how Qdrant is used in this project."

    client = None

    try:
        print("Running Tavily web search...")
        web_context = TavilyWebSearch()(user_prompt, max_results=WEB_TOP_K)
        print("Collected web evidence.")

        print(f"Persisting Qdrant DB in: {QDRANT_LOCAL_PATH}")
        rag_context = QdrantRAG()(user_prompt, PDF_DATA_DIR)

        combined_prompt = (
            "Answer using both contexts below: Web context from Tavily and PDF RAG context from Qdrant. "
            "If they disagree, explain the difference briefly. Keep the answer under 120 words.\n\n"
            f"Web context (Tavily):\n{web_context}\n\n"
            f"RAG context (Qdrant):\n{rag_context}\n\n"
            f"Question:\n{user_prompt}"
        )

        print("Generating final answer with OpenAI API...")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        print("Model response (streaming):\n")
        output = openai_infer(
            combined_prompt,
            model=OPENAI_MODEL,
            api_key=openai_api_key,
            max_tokens=MAX_OUTPUT_TOKENS,
            stream=True,
        )
        print("\nGeneration done")
    except Exception as err:
        print(f"Error: {err}")
    finally:
        if client is not None:
            client.close()
