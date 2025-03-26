import chromadb
import json
# setup Chroma in-memory, for easy prototyping. Can add persistence easily!
client = chromadb.Client()

# Create collection. get_collection, get_or_create_collection, delete_collection also available!
collection = client.create_collection("all-my-documents")

# Add docs to the collection. Can also update and delete. Row-based API coming soon!
# Load documents from JSON files
with open('ChromaDB/Data/C0000578F.json', 'r') as f:
    document1 = json.load(f)
# with open('document2.json', 'r') as f:
#     document2 = json.load(f)

collection.add(
    documents=[document1['text']], # assuming the JSON has a 'text' field
    metadatas=[document1['metadata']], # assuming the JSON has a 'metadata' field
    ids=[document1['id']], # assuming the JSON has an 'id' field
)

# Query/search 2 most similar results. You can also .get by id
results = collection.query(
    query_texts=["This is a query document"],
    n_results=2,
    # where={"metadata_field": "is_equal_to_this"}, # optional filter
    # where_document={"$contains":"search_string"}  # optional filter
)