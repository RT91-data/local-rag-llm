import os
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Settings
PDF_FILE_PATH = "my_data.pdf"

def main():
    if not os.path.exists(PDF_FILE_PATH):
        print(f"Error: Put '{PDF_FILE_PATH}' in this folder first!")
        return

    print("--- 1. Loading & Splitting PDF ---")
    loader = PyPDFLoader(PDF_FILE_PATH)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)

    print("--- 2. Creating Library (FAISS) ---")
    # Note: Ensure you have run 'ollama pull nomic-embed-text'
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    print("--- 3. Setting up AI ---")
    # Using llama3 or phi3 depending on your preference
    llm = ChatOllama(model="llama3") 
    
    # HYBRID PROMPT: Allows general chat + PDF analysis
    template = """You are a helpful AI assistant.
    
    1. If the user greets you or asks general questions (e.g., "How are you?"), answer naturally using your general knowledge.
    2. If the user asks about the document, use the context provided below to answer accurately.
    3. If the answer isn't in the context but relates to the document's topic, you may use your general knowledge but mention that the document doesn't explicitly state it.

    CONTEXT:
    {context}
    
    QUESTION: 
    {question}
    
    ANSWER:"""
    
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("--- READY! ---")
    while True:
        query = input("\nAsk (or 'exit'): ")
        if query.lower() == 'exit': break
        
        print("Thinking...")
        # We use invoke to get the full answer for the console version
        response = rag_chain.invoke(query)
        print("\nAI Answer:", response)

if __name__ == "__main__":
    main()