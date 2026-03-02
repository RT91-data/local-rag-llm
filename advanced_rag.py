import os
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_transformers import LongContextReorder
from langchain_core.messages import HumanMessage, AIMessage

# --- Configuration ---
DATA_PATH = "my_documents/"
INDEX_PATH = "faiss_index_storage"
MODEL_NAME = "llama3.1"

# 1. Initialize AI Models
print("--- Initializing AI Models ---")
embeddings = OllamaEmbeddings(model=MODEL_NAME)
llm = ChatOllama(model=MODEL_NAME, temperature=0.1)
reorder_transformer = LongContextReorder()

def get_hybrid_retriever():
    pdf_files = [f for f in os.listdir(DATA_PATH) if f.endswith(".pdf")]
    if not pdf_files:
        print("Error: No PDFs found.")
        return None

    all_docs = []
    # Optimization: Using larger chunks (800) for multi-doc context
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)

    print(f"Indexing {len(pdf_files)} files...")
    for f in pdf_files:
        loader = PyPDFLoader(os.path.join(DATA_PATH, f))
        all_docs.extend(loader.load())

    chunks = text_splitter.split_documents(all_docs)

    # Vector Search (FAISS)
    if os.path.exists(INDEX_PATH):
        vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(INDEX_PATH)
    
    # Increase k to 6 for better coverage across 5+ PDFs
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    # Keyword Search (BM25)
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 6

    # Hybrid Ensemble
    return EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever], 
        weights=[0.4, 0.6] # Vectors weighted slightly higher for multi-doc meaning
    )

if __name__ == "__main__":
    retriever = get_hybrid_retriever()
    chat_history = [] 

    print("\n--- ADVANCED MULTI-DOC RAG READY ---")
    while True:
        query = input("\nUser: ")
        if query.lower() in ['exit', 'quit']: break

        # 1. Retrieve & Reorder (Fixes "Lost in the Middle")
        raw_docs = retriever.invoke(query)
        reordered_docs = reorder_transformer.transform_documents(raw_docs)
        
        # 2. Build Context with Provenance
        context_list = []
        for i, d in enumerate(reordered_docs):
            file = os.path.basename(d.metadata.get('source', 'Unknown'))
            page = d.metadata.get('page', 'Unknown')
            context_list.append(f"SOURCE {i+1} [{file}, Pg {page}]:\n{d.page_content}")
        
        context = "\n\n".join(context_list)
        history_str = "\n".join([f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content}" for m in chat_history[-4:]])

        # 3. High-Precision Prompt
        prompt = f"""
        Instructions: Use the Context and History to answer. 
        If info is in multiple files, compare them. Cite File and Page for every fact.
        If the answer isn't in the Context, say 'I cannot find this in the documents.'

        Context:
        {context}

        History:
        {history_str}

        Question: {query}
        """

        print("Thinking...")
        response = llm.invoke(prompt)
        
        # Update History
        chat_history.append(HumanMessage(content=query))
        chat_history.append(AIMessage(content=response.content))

        print(f"\nAI: {response.content}")