__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
from matplotlib import pyplot as plt

# Create database file at folder "my_vectordb" or load into client if exists.
chroma_client = chromadb.PersistentClient(path="my_vectordb")

# Instantiate image loader helper.
image_loader = ImageLoader()

# Instantiate multimodal embedding function.
multimodal_ef = OpenCLIPEmbeddingFunction()

# Create the collection, aka vector database. Or, if database already exist, then use it. Specify the model that we want to use to do the embedding.
multimodal_db = chroma_client.get_or_create_collection(name="multimodal_db", embedding_function=multimodal_ef, data_loader=image_loader)

multimodal_db.add(
    ids = ['0', '1', '2', '3', '4'],
    uris = ['images/Stephen_Curry_2.jpg', 'images/christian-pulisic.jpeg', 'images/george-kittle.jpg', 'images/heliot-ramos.jpeg', 'images/macklin-celebrini.jpeg'],
)

multimodal_db.count()

# Simple function to print the results of a query.
# The 'results' is a dict {ids, distances, data, ...}
# Each item in the dict is a 2d list.
def print_query_results(query_list: list, query_results: dict)->None:
    result_count = len(query_results['ids'][0])

    for i in range(len(query_list)):
        print(f'Results for query: {query_list[i]}')

        for j in range(result_count):
            id       = query_results["ids"][i][j]
            distance = query_results['distances'][i][j]
            data     = query_results['data'][i][j]
            document = query_results['documents'][i][j]
            metadata = query_results['metadatas'][i][j]
            uri      = query_results['uris'][i][j]

            print(f'id: {id}, distance: {distance}, metadata: {metadata}, document: {document}')

            # Display image, the physical file must exist at URI.
            # (ImageLoader loads the image from file)
            print(f'data: {uri}')
            plt.imshow(data)
            plt.axis("off")
            plt.show()

query_texts = ['Lakers']

query_result = multimodal_db.query(
    query_texts = query_texts,
    n_results = 2,
    include = ['documents','distances','metadatas','data', 'uris'],
)

print_query_results(query_texts, query_result)

# Saves images in chroma and then gets their embeddings from CLIP
# Converts query text into CLIP embeddings and finds similar images
