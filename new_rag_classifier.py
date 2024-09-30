__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import numpy as np
from PIL import Image
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv
import torch
from transformers import ViTModel, ViTImageProcessor
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

# Retrieve the OpenAI API key from environment variables
api_key = os.getenv("OPENAI_API_KEY")
if api_key is None:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

os.environ["OPENAI_API_KEY"] = api_key


# Initialize ViT model and feature extractor
model_name = 'google/vit-base-patch16-224-in21k'
vit_model = ViTModel.from_pretrained(model_name)
feature_extractor = ViTImageProcessor.from_pretrained(model_name)

# Define your image stash
image_stash = {
    'images/spaldingbball.jpeg': 'Class1',
    'images/wilsonbball.jpeg': 'Class1',
    'images/wnbabball.jpeg': 'Class1',
    'images/fibabball.jpeg': 'Class1',
    'images/streetphantom.jpeg': 'Class1',
    'images/blankbaseball.jpg': 'Class2',
    'images/rawlingsbaseball.jpg': 'Class2',
    'images/wilsonbaseball.jpg': 'Class2',
    'images/nflfball.jpeg': 'Class3',
    'images/goldlogofball.jpeg': 'Class3',
    'images/wilsonncaafball.jpeg': 'Class3',
    'images/umichfball.jpeg': 'Class3',
}

# Function to preprocess an image
def preprocess_image(image_path: str) -> dict:
    """Preprocess an image and prepare it for the model."""
    image = Image.open(image_path)
    inputs = feature_extractor(images=image, return_tensors="pt")
    return inputs

# Function to compute image embeddings
def compute_embeddings(inputs: dict) -> np.ndarray:
    """Compute embeddings for a given image."""
    with torch.no_grad():
        outputs = vit_model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
    return embeddings.numpy()

# Add images to the database
def add_images_to_database(image_stash: dict):
    """Add images and their class labels to the database."""
    all_embeddings = []
    all_classes = []
    for uri, class_name in image_stash.items():
        inputs = preprocess_image(uri)
        embeddings = compute_embeddings(inputs)
        all_embeddings.append(embeddings)
        all_classes.append(class_name)
    return all_embeddings, all_classes

def query_image_class(query_image_path: str, database_embeddings: list, database_classes: list) -> tuple:
    """Query the class of the input image using similarity to the database."""
    query_inputs = preprocess_image(query_image_path)
    query_embeddings = compute_embeddings(query_inputs)

    # Calculate cosine similarity
    similarities = [
        cosine_similarity(query_embeddings, db_embedding)[0][0] for db_embedding in database_embeddings
    ]


    # Find the index of the most similar image
    best_match_index = np.argmax(similarities)

    # Get the best similarity score
    best_similarity_score = similarities[best_match_index]

    return database_classes[best_match_index], best_similarity_score

'''
# Query image class
def query_image_class(query_image_path: str, database_embeddings: list, database_classes: list) -> str:
    """Query the class of the input image using similarity to the database."""
    query_inputs = preprocess_image(query_image_path)
    query_embeddings = compute_embeddings(query_inputs)
    similarities = [cosine_similarity(query_embeddings, db_embedding)[0][0] for db_embedding in database_embeddings]
    best_match_index = np.argmax(similarities)
    return database_classes[best_match_index]
'''
# Function to get an explanation for the predicted class
def get_class_explanation(predicted_class: str, query_image_path: str, image_stash: dict) -> str:
    """Generate an explanation for why the queried image belongs to the predicted class."""
    explanation = f"The image at {query_image_path} is classified as '{predicted_class}' based on its similarity to the following images in the same class: "
    similar_images = [uri for uri, cls in image_stash.items() if cls == predicted_class]
    explanation += ", ".join(similar_images)
    return explanation

# Add images to the database
database_embeddings, database_classes = add_images_to_database(image_stash)

if __name__ == "__main__":
    # Example query image
    query_image_path = 'images/partofball.jpeg'

    # Predict class of the query image and get similarity score
    predicted_class, best_similarity_score = query_image_class(query_image_path, database_embeddings, database_classes)

    # Print similarity score
    print(f"Best similarity score: {best_similarity_score:.4f}")

    if best_similarity_score < 0.5:
        print("This image does not belong to any of the classes.")
    else:
        # Get explanation for the predicted class
        class_explanation = get_class_explanation(predicted_class, query_image_path, image_stash)
        print(f"Predicted class: {predicted_class}")
        print(f"Class explanation: {class_explanation}")

# Adds images into a chroma on a class basis. Then queries an image using chroma  and get the matched class and similarity.
