__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

class MyEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = self.model.encode(input, convert_to_tensor=False)
        embeddings_list = embeddings.tolist()
        return embeddings_list

# Initialize ChromaDB client and custom embedding function
client = chromadb.Client()
embedder = MyEmbeddingFunction()

# Function to add a document to ChromaDB
def add_document(collection_name, id, title, content):
    embedding = embedder([content])[0]  # Generate embedding for the content
    # Create or get the collection
    collection = client.get_or_create_collection(name=collection_name, embedding_function=embedder)
    collection.add(ids=[id], embeddings=[embedding], metadatas=[{'title': title, 'content': content}])

# Function to query documents from ChromaDB
def query_document(collection_name, query_text):
    query_embedding = embedder([query_text])[0]
    collection = client.get_collection(name=collection_name)
    num_elements = collection.count()
    n_result = min(5, num_elements)
    results = collection.query(query_embeddings=[query_embedding], n_results=n_result)
    return results

# Example usage
collection_name = "my_collection"
add_document(collection_name, "1", "Sample Title", "This is a sample content for the document.")
results = query_document(collection_name, "sample content")
print(results)

# creates collection with custom embeddings and queries
