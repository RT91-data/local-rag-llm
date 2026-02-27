import streamlit as st
import os
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- Page Config ---
st.set_page_config(page_title="AI Document Assistant", layout="wide")
st.title("📄 ERP Consultant's PDF Chatbot")
st.markdown("Upload a PDF and ask questions using local AI (Ollama).")

# --- Sidebar for Settings ---
with st.sidebar:
    st.header("Settings")
    model_choice = st.selectbox("Select Model", ["phi3", "llama3"], index=0)
    chunk_size = st.slider("Chunk Size", 500, 2000, 1000)
    uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

# --- Helper Functions ---
def process_pdf(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    return vectorstore.as_retriever()

# --- Main App Logic ---
if uploaded_file:
    # Save uploaded file temporarily (like a staging table)
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    with st.status("Analyzing document...", expanded=True) as status:
        retriever = process_pdf("temp.pdf")
        status.update(label="Analysis complete!", state="complete", expanded=False)

    # Initialize LLM
    llm = ChatOllama(model=model_choice)
    template = """Answer based ONLY on the context: {context}\nQuestion: {question}"""
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )

    # Chat Interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_query := st.chat_input("Ask me anything about the document..."):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            response = st.write_stream(chain.stream(user_query)) # This adds streaming!
        st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.info("Please upload a PDF file to begin.")