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

# Function to preprocess images
def preprocess_image(image_path):
    image = Image.open(image_path)
    inputs = feature_extractor(images=image, return_tensors="pt")

    # Extract positional information
    patch_positions = extract_patch_positions(image)

    return inputs, patch_positions

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

# Function to extract patch positions
def extract_patch_positions(image):
    # Define how patches are extracted from the image
    # Return positions as a list of tuples (x, y, width, height)
    patch_size = 16  # Example patch size, adjust as needed
    positions = []
    width, height = image.size

    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):
            positions.append((x, y, patch_size, patch_size))

    return positions

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
def add_images_to_database(image_uris, repeat=1, use_patch_embeddings=True):
    for uri_idx, uri in enumerate(image_uris):
        for repeat_idx in range(repeat):
            try:
                # Preprocess image
                inputs, patch_positions = preprocess_image(uri)

                # Get embeddings (768 dimensions)
                embeddings = vit_embedding_fn_768(inputs)

                # Calculate unique ID for the whole image
                unique_id = str(uri_idx + repeat_idx * len(image_uris))

                # Add patches if use_patch_embeddings is True
                if use_patch_embeddings:
                    # Compute patch embeddings
                    patch_embeddings = compute_embeddings(inputs, extract_patches=True)
                    patch_ids = [f"{unique_id}_patch_{i}" for i in range(patch_embeddings.shape[1])]

                    # Add patches to the database
                    for patch_id, patch_embedding in zip(patch_ids, patch_embeddings[0]):
                        multimodal_db_768.add(
                            ids=[patch_id],
                            embeddings=[patch_embedding.tolist()],
                            uris=[uri]
                        )
                    # Store patch positions separately
                    position_store.extend(zip(patch_ids, patch_positions))

                # Add image to the database with dimensionality 768
                multimodal_db_768.add(
                    ids=[unique_id],
                    embeddings=[embeddings.tolist()[0]],
                    uris=[uri]
                )

                print(f"Added image {uri} with ID {unique_id} and patches to database with dimensionality 768.")
            except Exception as e:
                print(f"Error adding image {uri}: {e}")

# Function to query similar patches
def query_similar_patches(query_image_path, image_uris, top_k=5, repeat=1, id_range=None):
    try:
        # Preprocess query image
        query_inputs, query_positions = preprocess_image(query_image_path)

        # Get embeddings for query image
        query_embeddings = compute_embeddings(query_inputs, extract_patches=True)

        # Check if query embeddings are empty
        if query_embeddings.shape[0] == 0:
            print(f"No patch embeddings found for query image {query_image_path}.")
            return []

        # Retrieve all patch embeddings and URIs from the database
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
                    inputs, patch_positions = preprocess_image(image_path)
                    patch_embeddings = compute_embeddings(inputs, extract_patches=True)

                    # Check if patch embeddings are empty
                    if patch_embeddings.shape[0] == 0:
                        print(f"No patch embeddings found for image {image_path}.")
                        continue

                    # Append embeddings, URIs, and IDs
                    all_embeddings.extend(patch_embeddings[0])
                    all_uris.extend([image_path] * patch_embeddings.shape[1])
                    all_ids.extend([f"{unique_id}_patch_{i}" for i in range(patch_embeddings.shape[1])])
                except Exception as e:
                    print(f"Error retrieving embeddings for image at index {uri_idx}, repeat {repeat_idx}: {e}")

        # Check if there are any embeddings to compare
        if len(all_embeddings) == 0:
            print("No patch embeddings found in the database.")
            return []

        # Convert lists to numpy arrays
        all_embeddings = np.array(all_embeddings)
        query_embeddings = np.array(query_embeddings[0])

        # Print shapes for debugging
        print(f"query_embeddings shape: {query_embeddings.shape}")
        print(f"all_embeddings shape: {all_embeddings.shape}")

        # Ensure embeddings are reshaped to 2D arrays
        if query_embeddings.ndim > 2:
            query_embeddings = query_embeddings.reshape(query_embeddings.shape[0], -1)
        if all_embeddings.ndim > 2:
            all_embeddings = all_embeddings.reshape(all_embeddings.shape[0], -1)

        # Compute cosine similarity between query patches and all patches
        similarity_scores = cosine_similarity(query_embeddings, all_embeddings)

        # Ensure similarity scores are 2D
        if similarity_scores.ndim != 2:
            print(f"Similarity scores shape is incorrect: {similarity_scores.shape}. Expected 2D.")
            return []

        # Find indices of top-k similar patches
        similar_indices = np.argsort(similarity_scores.flatten())[-top_k:][::-1]

        # Retrieve URIs, IDs, distances, and positions of top-k similar patches
        similar_results = []
        for idx in similar_indices:
            patch_index = idx % similarity_scores.shape[1]
            uri = all_uris[patch_index]
            unique_id = all_ids[patch_index]
            # Find the position from the position store
            position = next((pos for pid, pos in position_store if pid == unique_id), None)
            distance = similarity_scores.flatten()[idx]
            similar_results.append({'id': unique_id, 'uri': uri, 'distance': distance, 'position': position})

        return similar_results

    except Exception as e:
        print(f"Error querying similar patches for {query_image_path}: {e}")
        return []

# Function to print image IDs
def print_image_ids(image_uris, repeat=1):
    for uri_idx, uri in enumerate(image_uris):
        for repeat_idx in range(repeat):
            unique_id = uri_idx + repeat_idx * len(image_uris)
            print(f"ID: {unique_id}, URI: {uri}")

# Position storage for patches
position_store = []

# Example usage to add images to the database
image_uris = [
    'images/fiba_superimpose1.jpeg',
    'images/fiba_superimpose2.jpeg',
    'images/fiba_superimpose3.jpeg',
    'images/group-bball3.jpeg'
]

add_images_to_database(image_uris)

# Print image IDs
print_image_ids(image_uris)

# Query similar patches
query_image_path = 'images/partofball.jpeg'
similar_patches = query_similar_patches(query_image_path, image_uris, top_k=5, id_range=None)

# Print similar patches
print(f"Similar patches for {query_image_path}:")
for idx, result in enumerate(similar_patches):
    print(f"{idx + 1}. ID: {result['id']}, URI: {result['uri']}, Distance: {result['distance']}, Position: {result['position']}")

# Adds images to database and queries for most similar patches of an image to the queried image.
