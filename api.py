from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from rag_pipeline import build_qa
import shutil, os

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.post("/ask")
async def ask(file: UploadFile = File(...), question: str = "What is this about?"):
    # Save uploaded file temporarily
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Run RAG pipeline
    qa = build_qa(temp_path)
    answer = qa.run(question)

    # Cleanup
    os.remove(temp_path)
    return {"answer": answer}

@app.get("/")
def root():
    return {"status": "RAG API is running"}