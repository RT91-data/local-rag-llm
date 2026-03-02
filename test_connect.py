import os
from langchain_ollama import OllamaEmbeddings

print("Step 1: Script started...")

# Try a very simple connection
try:
    print("Step 2: Connecting to Ollama (llama3.1)...")
    embeddings = OllamaEmbeddings(model="llama3.1")
    
    # This is the 'ping' to the model
    print("Step 3: Sending a test word to see if AI is awake...")
    test_vector = embeddings.embed_query("Hello")
    
    print(f"SUCCESS! Vector generated with length: {len(test_vector)}")
    print("Your environment is working. The issue is likely with the PDF files or FAISS.")

except Exception as e:
    print(f"!!! FAILED: {str(e)}")
    print("\nTroubleshooting:")
    print("1. Is Ollama running in your taskbar?")
    print("2. Can you run 'ollama run llama3.1' in a separate terminal?")