import os
import mlflow
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

load_dotenv()

def ask(pdf_path, question):
    # Load PDF
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # Split
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # Embed
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Find relevant chunks
    relevant = vectorstore.similarity_search(question, k=3)
    context = "\n".join([d.page_content for d in relevant])

    # Ask LLM
    llm = ChatGroq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))
    response = llm.invoke(f"Context:\n{context}\n\nQuestion: {question}")
    return response.content

if __name__ == "__main__":
    mlflow.set_experiment("RAG-experiments")
    with mlflow.start_run():
        mlflow.log_param("chunk_size", 500)
        mlflow.log_param("model", "llama3-8b-8192")
        answer = ask("sample.pdf", "What is this document about?")
        print("Answer:", answer)