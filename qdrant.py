
# Qdrant RAG utility imports and config


import hashlib
from pathlib import Path
from typing import Iterable
from fastembed import TextEmbedding
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from config.config import load_config

# Load config from TOML using Pydantic
config = load_config()
qdrant_cfg = config.qdrant
QDRANT_COLLECTION = qdrant_cfg.collection
PDF_DATA_DIR = qdrant_cfg.pdf_data_dir
QDRANT_LOCAL_PATH = qdrant_cfg.local_path
EMBEDDING_MODEL = qdrant_cfg.embedding_model
CHUNK_SIZE = qdrant_cfg.chunk_size
CHUNK_OVERLAP = qdrant_cfg.chunk_overlap
TOP_K = qdrant_cfg.top_k

embedder = TextEmbedding(model_name=EMBEDDING_MODEL)





def get_qdrant_client() -> QdrantClient:
    QDRANT_LOCAL_PATH.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(QDRANT_LOCAL_PATH))


def iter_pdf_files(pdf_dir: Path) -> Iterable[Path]:
    return sorted(pdf_dir.glob("*.pdf"))


def extract_text_from_pdf(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages: list[str] = []
    for page in reader.pages:
        page_text = (page.extract_text() or "").strip()
        if page_text:
            pages.append(page_text)
    return "\n\n".join(pages)


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    normalized = " ".join(text.split())
    if not normalized:
        return []

    chunks: list[str] = []
    step = max(chunk_size - overlap, 1)
    start = 0
    while start < len(normalized):
        chunk = normalized[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


def chunk_id(source_path: str, chunk_index: int) -> str:
    key = f"{source_path}:{chunk_index}".encode("utf-8")
    return hashlib.md5(key).hexdigest()


def ensure_collection(client: QdrantClient, vector_size: int) -> None:
    collections = client.get_collections().collections
    existing = {collection.name for collection in collections}
    if QDRANT_COLLECTION in existing:
        return

    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )


def index_pdfs_into_qdrant(client: QdrantClient, pdf_dir: Path) -> int:
    pdf_files = list(iter_pdf_files(pdf_dir))
    if not pdf_files:
        return 0

    prepared: list[tuple[str, int, str]] = []
    for pdf_path in pdf_files:
        text = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        for idx, chunk in enumerate(chunks):
            prepared.append((pdf_path.name, idx, chunk))

    if not prepared:
        return 0

    vectors = [list(vec) for vec in embedder.embed([item[2] for item in prepared])]
    ensure_collection(client, vector_size=len(vectors[0]))

    points: list[PointStruct] = []
    for (source, idx, chunk), vector in zip(prepared, vectors, strict=True):
        points.append(
            PointStruct(
                id=chunk_id(source, idx),
                vector=vector,
                payload={
                    "source": source,
                    "chunk_index": idx,
                    "text": chunk,
                },
            )
        )

    client.upsert(collection_name=QDRANT_COLLECTION, points=points)
    return len(points)


def build_rag_context(client: QdrantClient, question: str, top_k: int = TOP_K) -> str:
    query_vector = list(next(embedder.embed([question])))
    results = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        limit=top_k,
    )

    if not results.points:
        return ""

    blocks: list[str] = []
    for i, point in enumerate(results.points, start=1):
        payload = point.payload or {}
        blocks.append(
            "\n".join(
                [
                    f"[doc_{i}]",
                    f"Source: {payload.get('source', 'unknown')}",
                    f"Chunk: {payload.get('chunk_index', 'n/a')}",
                    f"Score: {point.score}",
                    "Content:",
                    str(payload.get("text", "")),
                ]
            )
        )

    return "\n\n---\n\n".join(blocks)
