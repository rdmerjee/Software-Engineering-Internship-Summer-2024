import os
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from PIL import Image
import torch
from openai import OpenAI

# Load environment variables (optional, if using an API key stored in .env)
from dotenv import load_dotenv
load_dotenv()

# Set your OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
if api_key is None:
    raise ValueError("OPENAI_API_KEY environment variable not set.")
client = OpenAI(api_key=api_key)

# Initialize CLIP model, processor, and tokenizer
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

def generate_image_description(image_path):
    """Generate a description or features for an image using CLIP."""
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")
    
    # Generate image embeddings
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)

    # Generate a text description for the image (optional - based on model training or predefined prompts)
    description_prompt = "James Harden early in his career on the OKC Thunder"
    text_inputs = tokenizer(description_prompt, return_tensors="pt")
    
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
    
    # Return both image features and a descriptive text
    return image_features, description_prompt

def analyze_images_with_openai(descriptions):
    """Send descriptions to OpenAI for analysis."""
    prompt = "Analyze the following image descriptions:\n" + "\n".join(descriptions)
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an AI trained to analyze image descriptions and provide detailed insights."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300
    )
    
    # Access the message content correctly
    return response.choices[0].message.content.strip()

def main(image_directory):
    # Check if the directory exists
    if not os.path.isdir(image_directory):
        raise ValueError(f"The directory {image_directory} does not exist.")

    # Get a list of image paths
    image_paths = [os.path.join(image_directory, img) for img in os.listdir(image_directory) if img.endswith((".jpg", ".jpeg", ".png"))]

    if not image_paths:
        raise ValueError("No images found in the specified directory.")

    # Generate descriptions for each image
    descriptions = []
    for img_path in image_paths:
        image_features, description = generate_image_description(img_path)
        descriptions.append(description)
        print(f"Processed {img_path}")

    # Analyze the descriptions using OpenAI
    analysis = analyze_images_with_openai(descriptions)
    print("\nAnalysis Result:\n", analysis)

if __name__ == "__main__":
    # Set the directory containing your images
    image_directory = "./agent_images"

    # Run the analysis
    main(image_directory)

