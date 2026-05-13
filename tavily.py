from __future__ import annotations

import os
from typing import Any
import json
from urllib import error, request

from dotenv import load_dotenv


load_dotenv()


def tavily_web_search(query: str, max_results: int = 5) -> dict[str, Any]:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY is not set. Add it to .env to enable web context.")

    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": max_results,
    }
    req = request.Request(
        url="https://api.tavily.com/search",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Tavily API HTTP error {exc.code}: {details}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Tavily API network error: {exc.reason}") from exc


def normalize_tavily_results(tavily_response: dict[str, Any]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []

    for i, item in enumerate(tavily_response.get("results", []), start=1):
        content = str(item.get("content", "")).strip()
        if not content:
            continue

        evidence.append(
            {
                "source_id": f"web_{i}",
                "source_type": "web",
                "title": str(item.get("title", "")).strip(),
                "url": str(item.get("url", "")).strip(),
                "content": content,
                "score": item.get("score"),
            }
        )

    return evidence


def format_evidence_for_llm(evidence: list[dict[str, Any]]) -> str:
    blocks: list[str] = []

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