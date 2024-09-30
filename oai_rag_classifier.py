__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
from chromadb.utils.data_loaders import ImageLoader
from transformers import ViTModel, ViTImageProcessor
import torch
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KNeighborsClassifier
from openai import OpenAI
import os
import openai
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer

# Load environment variables (optional, if using an API key stored in .env)
from dotenv import load_dotenv
load_dotenv()

# Set your OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
if api_key is None:
    raise ValueError("OPENAI_API_KEY environment variable not set.")
client = OpenAI(api_key=api_key)

# Set the API key
openai.api_key = os.getenv("sk-proj-2FzVowxW5bL30HpoBAliT3BlbkFJ9WAH5UXwMmrhoqLcC25w")

# Create ChromaDB client
chroma_client = chromadb.PersistentClient(path="my_vectordb")

# Instantiate image loader helper
image_loader = ImageLoader()

# Instantiate ViT model and feature extractor
model_name = 'google/vit-base-patch16-224-in21k'
vit_model = ViTModel.from_pretrained(model_name)
feature_extractor = ViTImageProcessor.from_pretrained(model_name)

# Initialize CLIP model, processor, and tokenizer
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

# Example stash of images with labels
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

def generate_image_description(image_path):
    """Generate a description or features for an image using CLIP."""
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")

    # Generate image embeddings
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)

    # Generate a text description for the image (optional - based on model training or predefined prompts)
    description_prompt = "Lebron James in the prime of his career on the Cleveland Cavaliers"
    text_inputs = tokenizer(description_prompt, return_tensors="pt")

    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)

    # Return both image features and a descriptive text
    return image_features, description_prompt

def analyze_images_with_openai(descriptions):
    """Send descriptions to OpenAI for analysis."""
    prompt = "Analyze the following image descriptions and decide whether the description fits most with 2010's National Basketball Association or 2000's National Basketball Association:\n" + "\n".join(descriptions)

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an AI trained to analyze image descriptions and provide detailed insights."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300
    )

    return response.choices[0].message.content.strip()
'''
def get_class_explanation(class_name: str) -> str:
    """Generate an explanation for the predicted class using OpenAI's GPT."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Explain what a {class_name} is and its key characteristics."}
    ]

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # or another chat model
            messages=messages,
            max_tokens=150,  # Adjust as needed
            temperature=0.7  # Adjust as needed
        )
        explanation = response.choices[0].message['content'].strip()
        return explanation
    except Exception as e:
        print(f"Error getting class explanation: {e}")
        return "Explanation not available."
'''

# Add images to the database
database_embeddings, database_classes = add_images_to_database(image_stash)

# Query with a new image
query_image_path = 'images/lebron_james_2010s.jpeg'
predicted_class = query_image_class(query_image_path, database_embeddings, database_classes)

# Generate descriptions for each image
descriptions = []
for img in image_stash:
    image_features, description = generate_image_description(img)
    descriptions.append(description)
    print(f"Processed {img}")

# Get explanation for the predicted class
class_explanation = analyze_images_with_openai(descriptions)

print(f"Predicted class: {predicted_class}")
print(f"Class explanation: {class_explanation}")

# Adds images into a database and puts them in a class-basis. Then, queries an image and assigns the image to one of the classes and explains why.
