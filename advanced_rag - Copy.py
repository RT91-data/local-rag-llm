from __future__ import annotations
import pydantic
pydantic.v1_enabled = False # This tells LangChain to avoid the broken V1 layer
import os
import time
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever # Note: langchain_classic
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Configuration ---
DATA_PATH = "my_documents/"
FAISS_INDEX_PATH = "faiss_index_storage"
MODEL_NAME = "llama3.1" 
embeddings = OllamaEmbeddings(model=MODEL_NAME)

def get_hybrid_retriever():
    # 1. Gather all PDFs
    pdf_files = [f for f in os.listdir(DATA_PATH) if f.endswith(".pdf")]
    if not pdf_files:
        print("No PDFs found!")
        return None

    all_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)

    # 2. Load and Split ALL PDFs in the folder
    for pdf in pdf_files:
        print(f"Processing: {pdf}...")
        loader = PyPDFLoader(os.path.join(DATA_PATH, pdf))
        pages = loader.load()
        chunks = text_splitter.split_documents(pages)
        all_chunks.extend(chunks)

    # 3. Handle the Vector Store (FAISS)
    # If index exists, we load it. If you added NEW files, 
    # for now, the simplest way is to delete 'faiss_index_storage' 
    # and let it rebuild once with all files.
    if os.path.exists(FAISS_INDEX_PATH):
        print("--- Loading Existing Knowledge Base ---")
        vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        # Note: To properly 'add' without rebuilding, we'd use vectorstore.add_documents()
    else:
        print(f"--- Creating New Index for {len(pdf_files)} files... ---")
        vectorstore = FAISS.from_documents(all_chunks, embeddings)
        vectorstore.save_local(FAISS_INDEX_PATH)
    
    # 4. Keyword Search (BM25) needs all chunks to be effective
    bm25_retriever = BM25Retriever.from_documents(all_chunks)
    bm25_retriever.k = 3

    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    return EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever], 
        weights=[0.5, 0.5]
    )

# The rest of your main execution loop remains the same!