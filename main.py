"""
FastAPI backend for the portfolio AI assistant.

One real endpoint: POST /chat. Takes a visitor's question, embeds it,
retrieves the most relevant chunks of Jordi's project content from Pinecone,
and asks OpenAI to answer using only that retrieved context.

Deployed as its own Vercel project (Python runtime), separate from the
Next.js site. The site's chat widget calls this cross-origin.
"""

import json
import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pinecone import Pinecone
from pydantic import BaseModel

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "portfolio-ai-assistant")
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
TOP_K = 5

SYSTEM_PROMPT = """You are the AI assistant embedded on Jordi's AI engineering portfolio site.
Answer questions about Jordi's projects and background using ONLY the context provided below.
If the context doesn't cover what's being asked, say you don't have that information rather
than guessing. Keep answers concise and specific. Don't invent metrics, dates, or claims that
aren't in the context.

Each context chunk is labeled with a [slug] tag identifying its source. In used_slugs, list
ONLY the slugs of chunks that actually informed your answer. Retrieval sometimes returns
tangentially related or irrelevant chunks alongside the useful ones, don't cite a chunk just
because it was provided, only cite what you actually drew on."""

ANSWER_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "chat_answer",
        "schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "used_slugs": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["answer", "used_slugs"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

app = FastAPI(title="Portfolio AI Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS or ["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

_openai_client: OpenAI | None = None
_pinecone_index = None


def get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        if not OPENAI_API_KEY:
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


def get_pinecone_index():
    global _pinecone_index
    if _pinecone_index is None:
        if not PINECONE_API_KEY:
            raise HTTPException(status_code=500, detail="PINECONE_API_KEY not configured")
        pc = Pinecone(api_key=PINECONE_API_KEY)
        _pinecone_index = pc.Index(PINECONE_INDEX_NAME)
    return _pinecone_index


class ChatRequest(BaseModel):
    question: str


class Source(BaseModel):
    title: str
    slug: str
    url: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is required")

    client = get_openai_client()
    index = get_pinecone_index()

    # Embed the question the same way the content was embedded at index time.
    embedding = client.embeddings.create(model=EMBEDDING_MODEL, input=question).data[0].embedding

    # Retrieve the most relevant chunks.
    results = index.query(vector=embedding, top_k=TOP_K, include_metadata=True)
    matches = results.get("matches", [])

    if not matches:
        return ChatResponse(
            answer="I don't have anything indexed yet to answer that from. Check back once the content is loaded.",
            sources=[],
        )

    # Tag each chunk with its slug so the model can tell us which ones it actually used.
    context = "\n\n---\n\n".join(f"[{m['metadata']['slug']}]\n{m['metadata']['text']}" for m in matches)
    info_by_slug = {
        m["metadata"]["slug"]: {
            "title": m["metadata"].get("title", m["metadata"]["slug"]),
            "url": m["metadata"].get("url", "/"),
        }
        for m in matches
    }

    completion = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        temperature=0.2,
        response_format=ANSWER_SCHEMA,
    )

    try:
        parsed = json.loads(completion.choices[0].message.content)
        answer = parsed["answer"]
        used_slugs = parsed.get("used_slugs", [])
    except (json.JSONDecodeError, KeyError):
        # Fall back gracefully rather than 500ing on a malformed response.
        answer = completion.choices[0].message.content
        used_slugs = []

    sources = [
        Source(title=info_by_slug[slug]["title"], slug=slug, url=info_by_slug[slug]["url"])
        for slug in used_slugs
        if slug in info_by_slug
    ]

    return ChatResponse(answer=answer, sources=sources)
