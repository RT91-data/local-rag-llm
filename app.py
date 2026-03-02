import streamlit as st
import os
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- CONFIG & DIRECTORIES ---
st.set_page_config(page_title="Rupam's Permanent PA", layout="wide")
INDEX_DIR = "faiss_index_storage"
if not os.path.exists("temp_uploads"): os.makedirs("temp_uploads")

# --- INITIALIZE SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None

st.title("🤖 Rupam's Permanent Assistant")

# --- SIDEBAR: Controls ---
with st.sidebar:
    st.header("Storage & Settings")
    model_name = st.selectbox("Select Model", ["llama3.1", "phi3"], index=0)
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    
    if st.button("🚨 Wipe Data & Reset"):
        st.session_state.messages = []
        st.session_state.retriever = None
        if os.path.exists(INDEX_DIR):
            import shutil
            shutil.rmtree(INDEX_DIR)
        st.rerun()

# --- AI MODELS ---
llm = ChatOllama(model=model_name, temperature=0.1)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# --- PERSISTENT INDEXING LOGIC ---
# If index exists on disk but not in RAM, load it!
if st.session_state.retriever is None and os.path.exists(INDEX_DIR):
    with st.spinner("Loading saved index from disk..."):
        vectorstore = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
        # Rebuild Ensemble from the loaded vectorstore
        faiss_ret = vectorstore.as_retriever(search_kwargs={"k": 4})
        st.session_state.retriever = faiss_ret # Simplified for persistent reload
        st.toast("Welcome back! Previous PDFs loaded.")

# If user uploads NEW files
if uploaded_files and st.session_state.retriever is None:
    with st.status("Indexing new documents...", expanded=True) as status:
        all_splits = []
        for uploaded_file in uploaded_files:
            file_path = os.path.join("temp_uploads", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            loader = PyPDFLoader(file_path)
            splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(loader.load())
            all_splits.extend(splits)
        
        # Create and Save FAISS locally
        vectorstore = FAISS.from_documents(all_splits, embeddings)
        vectorstore.save_local(INDEX_DIR)
        
        # Create Hybrid Ensemble
        bm25 = BM25Retriever.from_documents(all_splits)
        faiss_ret = vectorstore.as_retriever(search_kwargs={"k": 4})
        st.session_state.retriever = EnsembleRetriever(retrievers=[bm25, faiss_ret], weights=[0.4, 0.6])
        status.update(label="Index saved to disk!", state="complete")

# --- CHAT UI ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if query := st.chat_input("Ask me about the documents..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"): st.markdown(query)

    with st.chat_message("assistant"):
        # 1. Retrieve Context
        context = ""
        if st.session_state.retriever:
            docs = st.session_state.retriever.invoke(query)
            context = "\n\n".join([f"Source: {d.metadata['source']}\n{d.page_content}" for d in docs])
        
        # 2. Prepare History String
        hist = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-5:-1]])
        
        # 3. Final RAG Prompt
        prompt = f"""Use the History and PDF Context to answer.
        HISTORY: {hist}
        CONTEXT: {context}
        QUESTION: {query}
        """
        
        response = st.write_stream(llm.stream(prompt))
        st.session_state.messages.append({"role": "assistant", "content": response})