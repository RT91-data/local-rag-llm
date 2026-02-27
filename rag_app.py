from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# 1. Load a PDF (Put any pdf in your folder and name it 'my_data.pdf')
loader = PyPDFLoader("my_data.pdf")
docs = loader.load()

# 2. Split the PDF into small chunks (AI can't read a whole book at once)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# 3. Create a "Vector Store" (The AI's temporary library)
# This will pull 'nomic-embed-text' if you don't have it: ollama pull nomic-embed-text
vectorstore = Chroma.from_documents(
    documents=splits, 
    embedding=OllamaEmbeddings(model="nomic-embed-text")
)

# 4. Set up the Brain (Llama3)
llm = ChatOllama(model="llama3")

# 5. Define how the AI should use the PDF context
system_prompt = (
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, say you don't know.\n\n{context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# 6. Build the Chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(vectorstore.as_retriever(), question_answer_chain)

# 7. Ask a question!
response = rag_chain.invoke({"input": "What is this document about?"})
print(response["answer"])