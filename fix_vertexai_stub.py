import os

path = os.path.join(
    "venv", "Lib", "site-packages",
    "langchain_community", "chat_models", "vertexai.py"
)

with open(path, "w") as f:
    f.write("from langchain_google_vertexai import ChatVertexAI\n")
    f.write("__all__ = ['ChatVertexAI']\n")

print("Stub written to:", path)