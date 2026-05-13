from __future__ import annotations


import os
from dotenv import load_dotenv
import openai
from config.config import load_config
from qdrant import get_qdrant_client, index_pdfs_into_qdrant, build_rag_context
from tavily import format_evidence_for_llm, normalize_tavily_results, tavily_web_search

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
        web_response = tavily_web_search(user_prompt, max_results=WEB_TOP_K)
        web_evidence = normalize_tavily_results(web_response)
        web_context = format_evidence_for_llm(web_evidence)
        print(f"Collected web evidence items: {len(web_evidence)}")

        print(f"Indexing PDFs from: {PDF_DATA_DIR}")
        print(f"Persisting Qdrant DB in: {QDRANT_LOCAL_PATH}")
        client = get_qdrant_client()
        indexed = index_pdfs_into_qdrant(client, PDF_DATA_DIR)
        print(f"Indexed/updated chunks: {indexed}")

        print("Retrieving context from Qdrant...")
        rag_context = build_rag_context(client, user_prompt, top_k=TOP_K)
        if not rag_context:
            raise RuntimeError("No RAG context found. Ensure PDFs exist in data/ and contain extractable text.")

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
