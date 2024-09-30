import os
import numpy as np
from PIL import Image
import torch
import openai
from transformers import ViTModel, ViTImageProcessor
from sklearn.metrics.pairwise import cosine_similarity

# Set the API key for OpenAI
openai_api_key = os.getenv("sk-proj-2FzVowxW5bL30HpoBAliT3BlbkFJ9WAH5UXwMmrhoqLcC25w")


# Instantiate ViT model and feature extractor
model_name = 'google/vit-base-patch16-224-in21k'
vit_model = ViTModel.from_pretrained(model_name)
feature_extractor = ViTImageProcessor.from_pretrained(model_name)

image_stash = {
    'images/allen_iverson_2000s.jpeg': 'Turn-of-Century',
    'images/tim_duncan_2000s.jpeg': 'Turn-of-Century',
    'images/kobe_bryant_2010s.jpeg': '3-Point Era',
    'images/tim_duncan_2010s.jpeg': '3-Point Era',
}

def preprocess_image(image_path: str) -> dict:
    """Preprocess an image and prepare it for the model."""
    image = Image.open(image_path)
    inputs = feature_extractor(images=image, return_tensors="pt")
    return inputs

def compute_embeddings(inputs: dict) -> np.ndarray:
    """Compute embeddings for a given image."""
    with torch.no_grad():
        outputs = vit_model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
    return embeddings.numpy()

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

def query_image_class(query_image_path: str, database_embeddings: list, database_classes: list) -> str:
    """Query the class of the input image using similarity to the database."""
    query_inputs = preprocess_image(query_image_path)
    query_embeddings = compute_embeddings(query_inputs)
    similarities = [cosine_similarity(query_embeddings, db_embedding)[0][0] for db_embedding in database_embeddings]
    best_match_index = np.argmax(similarities)
    return database_classes[best_match_index]



# Add images to the database
database_embeddings, database_classes = add_images_to_database(image_stash)

# Query with a new image
query_image_path = 'images/james_harden_okc_2010s.jpeg'
predicted_class = query_image_class(query_image_path, database_embeddings, database_classes)

print(f"Predicted class: {predicted_class}")
