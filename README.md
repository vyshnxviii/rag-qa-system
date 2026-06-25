# RAG Document Q&A System

Ask questions to any PDF using AI.

## What it does
- Upload any PDF and ask questions about it
- Uses FAISS vector search to find relevant chunks
- Sends context to Groq LLM (LLaMA 3.1) for answers
- Tracks experiments with MLflow

## Tech Stack
- LangChain + FAISS — vector search
- Groq API — free LLM inference
- HuggingFace Embeddings — all-MiniLM-L6-v2
- MLflow — experiment tracking
- FastAPI — REST API serving

## How to run
1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Add your Groq API key to `.env`
4. Run: `python rag_pipeline.py`

## Example output
> Q: What is this document about?
> A: This document is a resume highlighting skills in ML, backend engineering...
