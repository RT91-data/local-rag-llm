from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Initialize the local model
llm = ChatOllama(
    model="llama3",
    temperature=0.8,
)

# 2. Define a prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a witty assistant who gives concise answers."),
    ("user", "{topic}")
])

# 3. Create the chain using LCEL (LangChain Expression Language)
chain = prompt | llm | StrOutputParser()

# 4. Run the project
if __name__ == "__main__":
    user_input = input("What do you want to ask the AI? ")
    response = chain.invoke({"topic": user_input})
    print("\n--- AI Response ---")
    print(response)