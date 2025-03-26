import os
import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings

collection = "ChromaDB/Data"

# Directory where your persistent Chroma DB is stored
PERSIST_DIR = "./chroma_db"

# Load the SentenceTransformer model for embeddings
model = SentenceTransformer("all-mpnet-base-v2")

# Initialize the Chroma client with persistence enabled
client = chromadb.Client(Settings(persist_directory=PERSIST_DIR))

# Retrieve the collection that holds your image-transcription pairs
collection = client.get_or_create_collection(name="image_transcriptions")

def query_db(query_text, n_results=1):  # increased results
    query_embedding = model.encode(query_text).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["metadatas", "documents", "distances"]
    )
    return results

if __name__ == "__main__":
    query_text = input("Enter your query: ")
    results = query_db(query_text)
    print("Raw query results:", results)

    
    ids = results["ids"][0]
    metadatas = results["metadatas"][0]
    documents = results["documents"][0]
    distances = results["distances"][0]
    
    print("\nQuery Results:")
    for doc_id, metadata, document, distance in zip(ids, metadatas, documents, distances):
        print(f"Document ID: {doc_id}")
        print(f"Image Path: {metadata.get('image_path', 'N/A')}")
        print(f"Transcription: {metadata.get('transcription', 'N/A')}")
        print(f"Distance: {distance:.4f}")
        print("-" * 50)
