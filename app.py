import streamlit as st
import os
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="AI Hybrid Assistant", layout="wide")

st.title("🤖 Hybrid AI Assistant")
st.markdown("Ask me anything! Upload a PDF if you want me to analyze specific data.")

# --- SIDEBAR: Settings and Upload ---
with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox("Select Model", ["phi3", "llama3"], index=0)
    uploaded_file = st.file_uploader("Upload a PDF for context", type="pdf")

# --- INITIALIZE AI ---
llm = ChatOllama(model=model_name)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# --- HYBRID PROMPT ---
template = """You are a helpful AI assistant.
1. If the context below is empty or irrelevant, use your general knowledge.
2. If the context contains document data, prioritize that data to answer questions.

CONTEXT:
{context}

QUESTION: 
{question}
"""
prompt = ChatPromptTemplate.from_template(template)

# --- PDF PROCESSING LOGIC ---
retriever = None
if uploaded_file:
    with st.status("Processing PDF...", expanded=False) as status:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        loader = PyPDFLoader("temp.pdf")
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)
        
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()
        status.update(label="PDF Indexed!", state="complete")

# --- THE UNLOCKED CHAT INTERFACE ---
user_query = st.chat_input("Ask me a question...")

if user_query:
    with st.chat_message("user"):
        st.markdown(user_query)
    
    with st.chat_message("assistant"):
        # Helper to handle empty context when no PDF is uploaded
        def get_context(query):
            if retriever:
                docs = retriever.get_relevant_documents(query)
                return "\n\n".join(doc.page_content for doc in docs)
            return "No document uploaded. Answer using general knowledge."

        # Define the chain dynamically based on whether we have a retriever
        rag_chain = (
            {"context": lambda x: get_context(x), "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        # Stream the response
        st.write_stream(rag_chain.stream(user_query))