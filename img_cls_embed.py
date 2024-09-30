__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
from chromadb.utils.data_loaders import ImageLoader
from transformers import ViTModel, ViTFeatureExtractor
import torch
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Create ChromaDB client
chroma_client = chromadb.PersistentClient(path="my_vectordb")

# Instantiate image loader helper
image_loader = ImageLoader()

# Instantiate ViT model and feature extractor
model_name = 'google/vit-base-patch16-224-in21k'
vit_model = ViTModel.from_pretrained(model_name)
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)

from PIL import Image

# Function to preprocess images
def preprocess_image(image_path):
    image = Image.open(image_path)
    inputs = feature_extractor(images=image, return_tensors="pt")
    return inputs

# Define an embedding function class compatible with ChromaDB for 768 dimensions
class ViT768EmbeddingFunction:
    def __init__(self, model):
        self.model = model

    def __call__(self, input):
        with torch.no_grad():
            outputs = self.model(**input)
            embeddings = outputs.last_hidden_state[:, 0, :]  # Extract embeddings
        return embeddings.numpy()  # Convert to numpy array

# Create or get the new collection with ViT embedding function (dimensionality 768)
vit_embedding_fn_768 = ViT768EmbeddingFunction(vit_model)
multimodal_db_768 = chroma_client.get_or_create_collection(name="multimodal_db_768", embedding_function=vit_embedding_fn_768, data_loader=image_loader)

import torch
# Function to compute embeddings
def compute_embeddings(inputs, extract_patches=False):
    with torch.no_grad():
        outputs = vit_model(**inputs)
        if extract_patches:
            # Extract patch embeddings
            embeddings = outputs.last_hidden_state
        else:
            # Extract CLS token embedding
            embeddings = outputs.last_hidden_state[:, 0, :]
    return embeddings.numpy()

# Function to add images to the new database with dimensionality 768
def add_images_to_database(image_uris, repeat=5, use_patch_embeddings=True):
    for uri_idx, uri in enumerate(image_uris):
        for repeat_idx in range(repeat):
            try:
                # Preprocess image
                inputs = preprocess_image(uri)

                # Get embeddings (768 dimensions)
                embeddings = vit_embedding_fn_768(inputs)

                # Calculate unique ID for the whole image
                unique_id = str(uri_idx + repeat_idx * len(image_uris))

                # Add patches if use_patch_embeddings is True
                if use_patch_embeddings:
                    # Compute patch embeddings
                    patch_embeddings = compute_embeddings(inputs, extract_patches=True)
                    patch_ids = [f"{unique_id}_patch_{i}" for i in range(patch_embeddings.shape[0])]

                    # Add patches to the database
                    for patch_id, patch_embedding in zip(patch_ids, patch_embeddings):
                        multimodal_db_768.add(ids=[patch_id], embeddings=[patch_embedding.tolist()], uris=[uri])

                # Add image to the database with dimensionality 768
                multimodal_db_768.add(ids=[unique_id], embeddings=[embeddings.tolist()[0]], uris=[uri])

                print(f"Added image {uri} with ID {unique_id} and patches to database with dimensionality 768.")
            except Exception as e:
                print(f"Error adding image {uri}: {e}")

# Function to query similar images
def query_similar_images(query_image_path, image_uris, top_k=5, use_patch_embeddings=False, repeat=5, id_range=None):
    try:
        # Preprocess query image
        query_inputs = preprocess_image(query_image_path)

        # Get embeddings for query image
        query_embeddings = compute_embeddings(query_inputs, extract_patches=use_patch_embeddings)

        # Retrieve all embeddings and URIs from the database
        all_embeddings = []
        all_uris = []
        all_ids = []
        for uri_idx in range(len(image_uris)):
            for repeat_idx in range(repeat):
                unique_id = uri_idx + repeat_idx * len(image_uris)
                # Check if the current ID is within the specified range
                if id_range and (unique_id < id_range[0] or unique_id > id_range[1]):
                    continue
                try:
                    image_path = image_uris[uri_idx]
                    inputs = preprocess_image(image_path)
                    embeddings = compute_embeddings(inputs, extract_patches=use_patch_embeddings)
                    all_embeddings.append(embeddings)
                    all_uris.append(image_path)
                    all_ids.append(unique_id)
                except Exception as e:
                    print(f"Error retrieving embeddings for image at index {uri_idx}, repeat {repeat_idx}: {e}")

        # Convert to numpy arrays for cosine similarity calculation
        all_embeddings = np.array(all_embeddings)
        query_embeddings = np.array(query_embeddings)

        # Ensure embeddings are reshaped to 2D arrays
        if use_patch_embeddings:
            # Flatten patch embeddings
            query_embeddings = query_embeddings.reshape(query_embeddings.shape[0], -1)
            all_embeddings = all_embeddings.reshape(all_embeddings.shape[0], -1)
        else:
            query_embeddings = query_embeddings.reshape(1, -1)
            all_embeddings = all_embeddings.reshape(len(all_embeddings), -1)

        # Compute cosine similarity between query embeddings and all embeddings
        similarity_scores = cosine_similarity(query_embeddings, all_embeddings)

        # Find indices of top-k similar images
        similar_indices = np.argsort(similarity_scores[0])[-top_k:][::-1]

        # Retrieve URIs, IDs, and distances of top-k similar images
        similar_results = []
        for idx in similar_indices:
            uri = all_uris[idx]
            unique_id = all_ids[idx]
            distance = similarity_scores[0][idx]
            similar_results.append({'id': unique_id, 'uri': uri, 'distance': distance})

        return similar_results

    except Exception as e:
        print(f"Error querying similar images for {query_image_path}: {e}")
        return []

# Example usage to add images to the database
image_uris = [    
    'images/fiba_superimpose1.jpeg',
    'images/fiba_superimpose2.jpeg',
    'images/fiba_superimpose3.jpeg',
    'images/fibaball.jpeg',
    'images/group-bball3.jpeg'
]

add_images_to_database(image_uris)

# Example usage to print image IDs
def print_image_ids(image_uris, repeat=5):
    for uri_idx, uri in enumerate(image_uris):
        for repeat_idx in range(repeat):
            unique_id = uri_idx + repeat_idx * len(image_uris)
            print(f"ID: {unique_id}, URI: {uri}")

print_image_ids(image_uris)

# Example usage of query function with ID range
query_image_path = 'images/cutout1.jpeg'
similar_images = query_similar_images(query_image_path, image_uris, use_patch_embeddings=True, id_range=(0,4))

print(f"Similar images for {query_image_path}:")
for idx, result in enumerate(similar_images):
    print(f"{idx + 1}. ID: {result['id']}, URI: {result['uri']}, Distance: {result['distance']}")

# Adds images to the databse and queries for similar images and outputs similar CLS comparisons
