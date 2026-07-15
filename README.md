# Portfolio AI assistant

A RAG chatbot that answers questions about my project work, embedded directly on my portfolio site instead of living as a standalone demo.

**Stack:** Python, FastAPI, Pinecone, OpenAI (embeddings + generation) · **Paper:** [PDF](https://jordihako.com/papers/portfolio-ai-assistant.pdf) · **Video:** pending · **Write-up:** [jordihako.com](https://jordihako.com)

## What this is

A retrieval-augmented backend that indexes a set of source documents, answers questions grounded only in that content, and reports back which documents it actually used. Built to sit behind a chat widget on a live site rather than as a notebook demo, so it has to handle real constraints: serverless deployment, cross-origin calls from a browser, and answers that need to be reliably sourced, not just plausible-sounding.

## Quickstart

```bash
# install
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

# configure
cp .env.example .env   # fill in OPENAI_API_KEY, PINECONE_API_KEY, etc.

# build the index (run once, and again whenever source content changes)
python scripts/build_index.py

# run
uvicorn main:app --reload
```

## How it works

`scripts/build_index.py` reads source MDX files, chunks them, embeds each chunk with OpenAI, and upserts them into a Pinecone index along with metadata (title, slug, and the live URL each chunk maps back to). `main.py` exposes a single `/chat` endpoint: it embeds the incoming question, retrieves the closest chunks from Pinecone, and asks the model to answer using only that context. The model also reports which chunks it actually used, via a structured JSON response, so the sources returned to the caller reflect what informed the answer rather than everything retrieval happened to return.

## Result

Live and answering real questions on the deployed site. Indexes 14 source documents as 32 chunks. Retrieval and generation both verified against real questions, with source attribution confirmed accurate, no unrelated documents cited for on-topic questions.

## Limitations

No conversation memory across turns, each question is handled independently. No streaming, responses return in full rather than incrementally. Reindexing after source content changes is a manual script run, not automatic. No rate limiting yet, so cost exposure on the OpenAI side is uncapped under unexpected traffic.

## Links

- [Full write-up / case study](https://jordihako.com)
- [Paper (PDF)](https://jordihako.com/papers/portfolio-ai-assistant.pdf)
- [Video walkthrough](pending)
- [LinkedIn post](pending)
