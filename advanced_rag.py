import os
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Configuration
DATA_PATH = "my_documents/"  # Put your PDFs in this folder
CHROMA_PATH = "chroma_db_storage" # This is where your 'brain' lives on disk
MODEL_NAME = "llama3.1" # Or whatever model you pulled in Ollama

# 2. Initialize the Brain (Embeddings)
# This turns text into numbers locally
embeddings = OllamaEmbeddings(model=MODEL_NAME)

def load_or_create_db():
    if os.path.exists(CHROMA_PATH):
        print("--- Loading Existing Knowledge Base ---")
        return Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    
    print("--- Creating New Knowledge Base ---")
    # Load PDFs from the folder
    loaders = [PyPDFLoader(os.path.join(DATA_PATH, f)) for f in os.listdir(DATA_PATH) if f.endswith('.pdf')]
    docs = []
    for loader in loaders:
        docs.extend(loader.load())
    
    # Advanced Chunking: Keep paragraphs together
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    
    # Create and save the DB to disk
    db = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_PATH)
    return db

# 3. Execution
db = load_or_create_db()
llm = ChatOllama(model=MODEL_NAME, temperature=0.1) # low temp for accuracy

# 4. Simple Query Logic
query = "What are the main risks mentioned in the document?"
docs = db.similarity_search(query, k=3) # Get top 3 relevant chunks
context = "\n".join([d.page_content for d in docs])

prompt = f"Use this context: {context}\n\nQuestion: {query}\nAnswer:"
response = llm.invoke(prompt)

print("\nAI Response:\n", response.content)