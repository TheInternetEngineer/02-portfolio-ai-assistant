"""
Reads the website's MDX content, chunks it, embeds each chunk with OpenAI,
and upserts everything into Pinecone.

Rerun this any time content/ changes on the site (new project published,
existing writeup edited). Not automatic yet, run by hand:

    python scripts/build_index.py

Reads WEBSITE_CONTENT_DIR from .env, defaults to ../01-portfolio-website/content,
which assumes both project folders sit side by side in ~/Documents/portfolio-projects/.
"""

import os
import re
import sys
from pathlib import Path

import frontmatter
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "portfolio-ai-assistant")
CONTENT_DIR = Path(os.environ.get("WEBSITE_CONTENT_DIR", "../01-portfolio-website/content"))

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536  # text-embedding-3-small's output size
CHUNK_SIZE = 1200  # characters, not tokens, kept simple on purpose
CHUNK_OVERLAP = 150


def strip_jsx(body: str) -> str:
    """Drop import lines and JSX component tags MDX allows inline, keep the prose."""
    body = re.sub(r"^import .*$", "", body, flags=re.MULTILINE)
    body = re.sub(r"<[^>]+>", "", body)
    return body.strip()


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap
    return [c.strip() for c in chunks if c.strip()]


def load_documents(content_dir: Path) -> list[dict]:
    """Walk content/ for .mdx files, return frontmatter + cleaned body per file."""
    docs = []
    for path in content_dir.rglob("*.mdx"):
        post = frontmatter.load(path)
        title = post.get("title", path.stem)
        body = strip_jsx(post.content)
        docs.append({"slug": path.stem, "title": title, "text": body})
    return docs


def main():
    if not OPENAI_API_KEY or not PINECONE_API_KEY:
        sys.exit("Set OPENAI_API_KEY and PINECONE_API_KEY in .env before running this.")

    if not CONTENT_DIR.exists():
        sys.exit(f"Content dir not found: {CONTENT_DIR}. Check WEBSITE_CONTENT_DIR in .env.")

    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    pc = Pinecone(api_key=PINECONE_API_KEY)

    if PINECONE_INDEX_NAME not in [i["name"] for i in pc.list_indexes()]:
        print(f"Creating Pinecone index '{PINECONE_INDEX_NAME}'...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSIONS,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    index = pc.Index(PINECONE_INDEX_NAME)

    docs = load_documents(CONTENT_DIR)
    print(f"Found {len(docs)} MDX files under {CONTENT_DIR}")

    vectors = []
    for doc in docs:
        chunks = chunk_text(doc["text"])
        for i, chunk in enumerate(chunks):
            embedding = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=chunk).data[0].embedding
            vectors.append(
                {
                    "id": f"{doc['slug']}-{i}",
                    "values": embedding,
                    "metadata": {"text": chunk, "title": doc["title"], "slug": doc["slug"]},
                }
            )

    print(f"Embedded {len(vectors)} chunks, upserting to Pinecone...")
    for i in range(0, len(vectors), 100):
        index.upsert(vectors=vectors[i : i + 100])

    print("Done.")


if __name__ == "__main__":
    main()
