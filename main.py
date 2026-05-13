from openai import OpenAI
import requests
from tavily import TavilyClient
from dotenv import load_dotenv
import os
from termcolor import colored
from typing import Any


# Load environment variables from .env file
load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Create an OpenAI client configured for the Ollama local server
client = OpenAI(
    base_url='http://localhost:11434/v1/',
    api_key='ollama',  # Required but ignored by Ollama
)

# Initialize the Tavily client with the API key from the environment
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)


def ollama_infer(prompt: str, model: str = "llama3.2") -> str:
    """Run a single inference call using the Ollama local server."""
    try:
        response = client.chat.completions.create(
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                }
            ],
            model=model,
        )
        return response.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"Ollama API error: {e}")


def tavily_web_search(query: str) -> str:
    """Fetch context from a web search to provide additional information for the Ollama model."""
    try:
        response = tavily_client.search(query)
        print(colored("Tavily Web Search Response:\n" + str(response), color="green"))
      
        # Assuming the response contains a 'results' field with snippets
        return response
    except Exception as e:
        raise RuntimeError(f"Error fetching context from Tavily: {e}")


def process_tavily_results(tavily_response: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert Tavily response into clean web evidence objects."""
    evidence = []

    for i, item in enumerate(tavily_response.get("results", []), start=1):
        content = item.get("content", "").strip()

        if not content:
            continue

        evidence.append(
            {
                "source_id": f"web_{i}",
                "source_type": "web",
                "title": item.get("title", "").strip(),
                "url": item.get("url", "").strip(),
                "content": content,
                "score": item.get("score", None),
            }
        )

    return evidence


def format_tavily_response(evidence: list[dict]) -> str:
    blocks = []

    for item in evidence:
        block = f"""
[{item["source_id"]}]
Source type: {item["source_type"]}
Title: {item["title"]}
URL: {item["url"]}
Search score: {item["score"]}
Content:
{item["content"]}
""".strip()

        blocks.append(block)

    return "\n\n---\n\n".join(blocks)


if __name__ == "__main__":
    model_name = "llama3.2"  # Ollama model name
    user_prompt = "Explain what inference means in one short paragraph."

    try:
        # Fetch additional context from the web
        response = tavily_web_search(user_prompt)
        processed_response = process_tavily_results(response)
        web_context = format_tavily_response(processed_response)

        # Combine the context with the user prompt
        combined_prompt = f"{web_context}\n\n{user_prompt}"

        # Run inference with the combined prompt
        output = ollama_infer(combined_prompt, model=model_name)
        print("\nModel response:\n")
        print(output)
    except RuntimeError as err:
        print(f"Error: {err}")
