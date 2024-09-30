__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
from chromadb.utils import embedding_functions
chroma_client = chromadb.Client()

collection = chroma_client.get_or_create_collection(name="my_collection")

collection.add(
        documents = [
            "This is a document about pineapple",
            "This is a document about oranges"
        ],
        ids = ["id1","id2"]
)

results = collection.query(
        query_texts = ["This is a query document about florida"], # Chroma will embed this for you
        n_results = 2 # how many results to return
)

print(results)

# python3.10 chromascript.py
# This adds sentences to the collection and queries.
# For Hawaii, it gets pineapple, and for Florida it gets oranges. Don't know why or how it comes to this conclusion.
