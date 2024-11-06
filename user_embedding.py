import torch
from transformers import CLIPProcessor, CLIPModel, pipeline
from PIL import Image
import requests
from io import BytesIO
import json
from datasets import load_dataset

# Load the OpenAI CLIP model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
model.to(device)

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

# Load the Hugging Face dataset
df = load_dataset("FronkonGames/steam-games-dataset")['train']

# Initialize a dictionary to store embeddings by user ID
user_embeddings = {}
with open("output.json", "r") as file:
    cleaned_user_games_dict = json.load(file)

# Iterate through the user games dictionary
for user_id, games in cleaned_user_games_dict.items():
    user_embeddings[user_id] = []

    for game_title in games:
        # Find a match in the Hugging Face dataset
        match = None
        for item in df:
            if game_title and item['Name'] and game_title.lower() in item['Name'].lower():
                match = item
                break

        if match:
            try:
                # Access game data
                image_url = match['Header image']
                about_game = match['About the game']

                # Summarize the "About the Game" text
                summarized_about_game = summarizer(
                    about_game, max_length=min(len(about_game), 70), min_length=5, do_sample=False
                )[0]['summary_text']

                # Embed the game title and summarized about game
                title_embedding = embed(image_url, game_title)[1]  # Only get the text embedding for the title
                about_embedding = embed(image_url, summarized_about_game)[1]  # Only get the text embedding for the about game
                image_embedding = embed(image_url, game_title)[0]  # Get the image embedding

                # Store the embeddings in the user dictionary
                user_embeddings[user_id].append({
                    'game_title': game_title,
                    'image_embedding': image_embedding.cpu(),
                    'title_embedding': title_embedding.cpu(),
                    'about_embedding': about_embedding.cpu(),
                })
            
            except Exception as e:
                print(f"Error processing game '{game_title}' for user {user_id}: {e}. Skipping.")

# Save the user embeddings to .pt file
torch.save(user_embeddings, 'user_game_embeddings.pt')

print("User game embeddings saved successfully.")
