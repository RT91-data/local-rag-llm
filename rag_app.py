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
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    print("--- 3. Setting up AI ---")
    llm = ChatOllama(model="llama3")
    
    template = """Answer the question based only on the following context:
    {context}
    
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # This is the "Modern Style" that avoids the error you have
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
        query = input("\nAsk your PDF a question (or 'exit'): ")
        if query.lower() == 'exit': break
        
        print("Searching and thinking...")
        print("\nAI Answer:", rag_chain.invoke(query))

if __name__ == "__main__":
    main()