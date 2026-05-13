import os, ssl, requests, pickle, shutil
from functools import partial
from flashrank import Ranker
import streamlit as st

# --- 1. NETWORK & SSL BYPASS ---
os.environ['CURL_CA_BUNDLE'] = ''
ssl._create_default_https_context = ssl._create_unverified_context
requests.get = partial(requests.get, verify=False)

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_community.document_transformers import LongContextReorder

# --- 4. CRITICAL REBUILD ---
FlashrankRerank.model_rebuild()

# --- 5. CONFIG & SESSION STATE ---
st.set_page_config(page_title="Rupam's Smart Data Assistant", layout="wide")
INDEX_DIR = "faiss_index_storage"
TEMP_DIR = "temp_uploads"

if not os.path.exists(TEMP_DIR): os.makedirs(TEMP_DIR)
if "messages" not in st.session_state: st.session_state.messages = []
if "retriever" not in st.session_state: st.session_state.retriever = None

# --- 6. MODELS ---
# We use temperature 0 for the rewriter to ensure it stays literal
llm = ChatOllama(model="llama3.1", temperature=0)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# --- 7. SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Data Controls")
    uploaded_files = st.file_uploader("Upload PDFs (Table Optimized)", type="pdf", accept_multiple_files=True)
    
    if st.button("🚨 Clear Everything"):
        st.session_state.messages = []
        st.session_state.retriever = None
        if os.path.exists(INDEX_DIR): shutil.rmtree(INDEX_DIR)
        if os.path.exists(TEMP_DIR): shutil.rmtree(TEMP_DIR)
        st.rerun()
    st.info("Status: Conversational Mode Active")

# --- 8. PERSISTENT LOAD ---
if st.session_state.retriever is None and os.path.exists(INDEX_DIR):
    if os.path.exists(os.path.join(INDEX_DIR, "index.faiss")):
        try:
            with st.spinner("Reloading brain..."):
                vs = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
                with open(os.path.join(INDEX_DIR, "splits.pkl"), "rb") as f:
                    stored_splits = pickle.load(f)
                bm25 = BM25Retriever.from_documents(stored_splits)
                st.session_state.retriever = EnsembleRetriever(
                    retrievers=[bm25, vs.as_retriever(search_kwargs={"k":15})], weights=[0.4, 0.6]
                )
        except Exception: pass

# --- 9. INDEXING LOGIC ---
if uploaded_files and st.session_state.retriever is None:
    if st.button("🚀 Start Data Extraction"):
        with st.status("Reading Tables...", expanded=True) as status:
            all_docs = []
            for file in uploaded_files:
                path = os.path.join(TEMP_DIR, file.name)
                with open(path, "wb") as f: f.write(file.getbuffer())
                all_docs.extend(PDFPlumberLoader(path).load())
            
            splits = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(all_docs)
            if not os.path.exists(INDEX_DIR): os.makedirs(INDEX_DIR)
            
            vs = FAISS.from_documents(splits, embeddings)
            vs.save_local(INDEX_DIR)
            with open(os.path.join(INDEX_DIR, "splits.pkl"), "wb") as f: pickle.dump(splits, f)
            
            bm25 = BM25Retriever.from_documents(splits)
            st.session_state.retriever = EnsembleRetriever(
                retrievers=[bm25, vs.as_retriever(search_kwargs={"k":15})], weights=[0.4, 0.6]
            )
            status.update(label="Ready!", state="complete")
            st.rerun()

# --- 10. SMART CHAT UI (The Brain) ---
st.title("🤖 Smart Conversational Assistant")

for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

if query := st.chat_input("Ask about your data..."):
    if st.session_state.retriever is None:
        st.error("Please upload and index PDFs first!"); st.stop()

    # --- THE MAGIC STEP: QUERY REWRITING ---
    # We look at the last 2 turns to see if this is a follow-up or a new topic
    chat_history = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-2:]])
    
    rewrite_prompt = f"""Conversation history:
    {chat_history}
    
    User Question: {query}
    
    Based on the history, if the question is a follow-up, re-write it to be a standalone search query. 
    If it is a new topic, just return the question as is. Do not add any preamble.
    Standalone Search Query:"""
    
    refined_query = llm.invoke(rewrite_prompt).content
    
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"): st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Stage 1: Retrieval with the refined query
            compressor = FlashrankRerank(top_n=10)
            retriever = ContextualCompressionRetriever(
                base_compressor=compressor, 
                base_retriever=st.session_state.retriever
            )
            
            # Fetch & Reorder
            docs = LongContextReorder().transform_documents(retriever.invoke(refined_query))
            context_text = "\n\n".join([f"Source: {d.metadata.get('source')}\n{d.page_content}" for d in docs])
            
            # Stage 2: Final Answer
            final_prompt = f"""You are a helpful AI assistant. 
            Answer using the provided context. If the question is a follow-up, use the history.
            If the answer is in a table, be very precise with numbers.
            
            HISTORY: {chat_history}
            CONTEXT: {context_text}
            QUESTION: {query}
            """
            
            response = st.write_stream(llm.stream(final_prompt))
            st.session_state.messages.append({"role": "assistant", "content": response})