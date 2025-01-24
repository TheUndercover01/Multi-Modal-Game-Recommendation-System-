
import torch
from transformers import CLIPProcessor, CLIPModel, pipeline
from PIL import Image
import requests
from io import BytesIO
from datasets import load_dataset
# Load the OpenAI CLIP model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
model.to(device)
print(1)
# Load the BART summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Function to load an image from a URL
def load_image(image_url):
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    return image

# Function to embed an image and text
def embed(image_url, text):
    # Load and preprocess the image
    image = load_image(image_url)
    inputs = processor(text=text, images=image, return_tensors="pt", padding=True).to(device)

    # Get image and text embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        image_embeddings = outputs.image_embeds 
        text_embeddings = outputs.text_embeds 

    # Normalize the embeddings
    image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
    text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

    return image_embeddings, text_embeddings

import pandas as pd

df=load_dataset("FronkonGames/steam-games-dataset")
# Initialize lists to store embeddings
image_embeddings_list = []
title_embeddings_list = []
about_embeddings_list = []
game_names_list=[]
i=20000
# Iterate through the dataset
while i<25000:
    i=i+1
    try:
        image_url = df['train']['Header image'][i]
        game_title = df['train']['Name'][i] 
        about_game = df['train']['About the game'][i]  

    # Summarize the "About the Game" text
        summarized_about_game = summarizer(about_game, max_length=min(len(about_game),70), min_length=5, do_sample=False)[0]['summary_text']

    # Embed the game title
        title_embedding = embed(image_url, game_title)[1]  # Only get the text embedding for the title

    # Embed the summarized "About the Game" text
        about_embedding = embed(image_url, summarized_about_game)[1]  # Only get the text embedding for the about game

    # Embed the image
        image_embedding = embed(image_url, game_title)[0]  # Get the image embedding

    # Append embeddings to lists
        image_embeddings_list.append(image_embedding.cpu())
        title_embeddings_list.append(title_embedding.cpu())
        about_embeddings_list.append(about_embedding.cpu()) 
        game_names_list.append(game_title)
	
    except Exception as e: 	
        print("error")
# Convert lists to tensors
all_image_embeddings = torch.cat(image_embeddings_list)
all_title_embeddings = torch.cat(title_embeddings_list)
all_about_embeddings = torch.cat(about_embeddings_list)

# Save the embeddings to .pt file
torch.save({
    'game_names': game_names_list,
    'image_embeddings': all_image_embeddings,
    'title_embeddings': all_title_embeddings,
    'about_embeddings': all_about_embeddings
}, 'game_embeddings-12.pt')

print("Embeddings saved successfully.")
