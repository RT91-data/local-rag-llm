import streamlit as st
import os
import shutil
import pdfplumber
from dotenv import load_dotenv
from anthropic import Anthropic
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

# --- LOAD ENV ---
load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# --- CONFIG ---
st.set_page_config(page_title="Rupam's AI Assistant", layout="wide")
INDEX_DIR = "faiss_index_storage"
UPLOAD_DIR = "temp_uploads"

if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# --- SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None

st.title("🤖 Rupam's AI Assistant")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Settings")
    model_name = st.selectbox(
        "Claude Model",
        ["claude-sonnet-4-20250514", "claude-haiku-4-5-20251001"],
        index=0
    )
    st.divider()
    st.header("📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDFs", type="pdf", accept_multiple_files=True
    )
    st.divider()
    if st.button("🚨 Wipe All Data & Reset"):
        st.session_state.messages = []
        st.session_state.retriever = None
        if os.path.exists(INDEX_DIR):
            shutil.rmtree(INDEX_DIR)
        if os.path.exists(UPLOAD_DIR):
            shutil.rmtree(UPLOAD_DIR)
            os.makedirs(UPLOAD_DIR)
        st.success("Reset complete!")
        st.rerun()

# --- CLAUDE CLIENT ---
@st.cache_resource
def load_claude():
    return Anthropic(api_key=ANTHROPIC_API_KEY)

# --- LOAD EMBEDDINGS (still use Ollama for embeddings - free & local) ---
@st.cache_resource
def load_embeddings():
    return OllamaEmbeddings(model="nomic-embed-text")

claude = load_claude()
embeddings = load_embeddings()

# --- SMART PDF LOADER ---
def load_pdf_with_tables(file_path):
    docs = []
    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            page_text = ""
            tables = page.extract_tables()
            if tables:
                for table in tables:
                    if table:
                        table_lines = []
                        for row in table:
                            cleaned = [
                                str(cell).strip() if cell else ""
                                for cell in row
                            ]
                            table_lines.append(" | ".join(cleaned))
                        page_text += "\n[TABLE]\n"
                        page_text += "\n".join(table_lines)
                        page_text += "\n[/TABLE]\n\n"
            text = page.extract_text()
            if text:
                page_text += text
            if page_text.strip():
                docs.append(Document(
                    page_content=page_text,
                    metadata={"source": file_path, "page": page_num + 1}
                ))
    return docs

# --- HELPER: Build retriever ---
def build_retriever(chunks):
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(INDEX_DIR)
    faiss_ret = vectorstore.as_retriever(search_kwargs={"k": 4})
    bm25_ret = BM25Retriever.from_documents(chunks)
    bm25_ret.k = 4
    return EnsembleRetriever(
        retrievers=[bm25_ret, faiss_ret],
        weights=[0.4, 0.6]
    )

# --- HELPER: Load chunks from disk ---
def load_chunks_from_disk():
    chunks = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )
    if os.path.exists(UPLOAD_DIR):
        for f in os.listdir(UPLOAD_DIR):
            if f.endswith(".pdf"):
                file_path = os.path.join(UPLOAD_DIR, f)
                docs = load_pdf_with_tables(file_path)
                chunks.extend(splitter.split_documents(docs))
    return chunks

# --- PERSISTENT INDEX LOAD ---
if st.session_state.retriever is None and os.path.exists(INDEX_DIR):
    with st.spinner("Loading your previous documents..."):
        try:
            vectorstore = FAISS.load_local(
                INDEX_DIR, embeddings,
                allow_dangerous_deserialization=True
            )
            chunks = load_chunks_from_disk()
            if chunks:
                faiss_ret = vectorstore.as_retriever(search_kwargs={"k": 4})
                bm25_ret = BM25Retriever.from_documents(chunks)
                bm25_ret.k = 4
                st.session_state.retriever = EnsembleRetriever(
                    retrievers=[bm25_ret, faiss_ret],
                    weights=[0.4, 0.6]
                )
            else:
                st.session_state.retriever = vectorstore.as_retriever(
                    search_kwargs={"k": 4}
                )
            st.toast("✅ Previous documents loaded!")
        except Exception as e:
            st.warning(f"Could not load previous index: {e}")

# --- NEW FILE UPLOAD ---
if uploaded_files:
    new_files = [
        f for f in uploaded_files
        if not os.path.exists(os.path.join(UPLOAD_DIR, f.name))
    ]
    if new_files:
        with st.status(f"Indexing {len(new_files)} new file(s)...", expanded=True) as status:
            all_chunks = load_chunks_from_disk()
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )
            for uploaded_file in new_files:
                file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.write(f"📄 Processing: {uploaded_file.name}")
                docs = load_pdf_with_tables(file_path)
                chunks = splitter.split_documents(docs)
                all_chunks.extend(chunks)
            st.session_state.retriever = build_retriever(all_chunks)
            status.update(label="✅ Documents indexed!", state="complete")

# --- CHAT UI ---
st.divider()

if st.session_state.retriever:
    pdf_count = len([
        f for f in os.listdir(UPLOAD_DIR) if f.endswith(".pdf")
    ]) if os.path.exists(UPLOAD_DIR) else 0
    st.caption(f"📚 {pdf_count} document(s) loaded | Model: {model_name}")
else:
    st.info("👆 Upload PDFs from the sidebar, or ask a general question below.")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if query := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        # 1. Retrieve context
        context = ""
        sources_text = ""
        if st.session_state.retriever:
            docs = st.session_state.retriever.invoke(query)
            context_parts = []
            for i, doc in enumerate(docs):
                source = os.path.basename(doc.metadata.get("source", "Unknown"))
                page = doc.metadata.get("page", "?")
                context_parts.append(
                    f"[Source {i+1}: {source}, Page {page}]\n{doc.page_content}"
                )
            context = "\n\n".join(context_parts)
            sources = list(set([
                f"{os.path.basename(d.metadata.get('source', 'Unknown'))} "
                f"(p.{d.metadata.get('page', '?')})"
                for d in docs
            ]))
            sources_text = "\n".join(sources)

        # 2. Build system prompt
        if context:
            system_prompt = """You are a precise and helpful document assistant.
Answer questions using ONLY the context provided.
If the answer is not in the context, say: "I cannot find this in the provided documents."
For table data, read each row and column carefully.
Always cite the source filename and page number for every fact."""
        else:
            system_prompt = "You are a helpful AI assistant. Answer accurately and concisely."

        # 3. Build messages for Claude API
        # Include conversation history
        api_messages = []
        for m in st.session_state.messages[-6:-1]:
            api_messages.append({
                "role": m["role"],
                "content": m["content"]
            })

        # Add current question with context
        user_content = f"{query}"
        if context:
            user_content = f"""DOCUMENT CONTEXT:
{context}

QUESTION: {query}"""

        api_messages.append({
            "role": "user",
            "content": user_content
        })

        # 4. Stream response from Claude
        response_text = ""
        placeholder = st.empty()

        with claude.messages.stream(
            model=model_name,
            max_tokens=1024,
            system=system_prompt,
            messages=api_messages
        ) as stream:
            for text in stream.text_stream:
                response_text += text
                placeholder.markdown(response_text + "▌")

        placeholder.markdown(response_text)

        # 5. Show sources
        if sources_text and context:
            with st.expander("📎 Sources used"):
                st.caption(sources_text)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response_text
    })