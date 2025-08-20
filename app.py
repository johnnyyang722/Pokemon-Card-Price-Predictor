import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

# ==============================
# Load trained model
# ==============================
from model_definition import PokemonPricePredictor  # Import your model class

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model (make sure input_dim matches your rarity encoding size)
model = PokemonPricePredictor(input_dim=7)  # adjust if your rarity one-hot is different
model.load_state_dict(torch.load("pokemon_price_predictor_augmented_finetuned.pt", map_location=device))
model.to(device)
model.eval()

# ==============================
# Rarity mapping
# ==============================
rarity_map = {
    "Common": 0,
    "Uncommon": 1,
    "Rare": 2,
    "Rare Holo": 3,
    "Rare Ultra": 4,
    "Rare Secret": 5,
    "Promo": 6
}
idx_to_rarity = {v: k for k, v in rarity_map.items()}

# ==============================
# Image preprocessing
# ==============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def preprocess_image(image_file):
    image = Image.open(image_file).convert("RGB")
    return transform(image).unsqueeze(0)  # add batch dimension


# ==============================
# Prediction function
# ==============================
def predict_price(image_tensor, rarity_str):
    rarity_idx = rarity_map[rarity_str]
    rarity_tensor = torch.tensor([rarity_idx], dtype=torch.long).to(device)

    with torch.no_grad():
        log_pred = model(image_tensor.to(device), rarity_tensor).item()
        pred_price = np.expm1(log_pred)  # reverse log1p transform
    return pred_price


# ==============================
# Streamlit UI
# ==============================
st.title("Pokémon Card Price Predictor")

st.write("Upload a Pokémon card image, select rarity, and enter the card name to predict market price.")

# Card name input
card_name_input = st.text_input("Enter Card Name (optional):")

# Rarity dropdown
rarity_choice = st.selectbox("Select Rarity:", list(rarity_map.keys()))

# Image upload
uploaded_file = st.file_uploader("Upload a card image:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Card Image", use_column_width=True)

    # If user left card name blank, default to filename
    if not card_name_input:
        card_name = os.path.splitext(uploaded_file.name)[0]
    else:
        card_name = card_name_input

    # Preprocess image
    image_tensor = preprocess_image(uploaded_file)

    # Predict price
    predicted_price = predict_price(image_tensor, rarity_choice)

    # Display results
    st.subheader("Prediction Results")
    st.write(f"**Card Name:** {card_name}")
    st.write(f"**Rarity:** {rarity_choice}")
    st.write(f"**Predicted Price:** ${predicted_price:.2f}")
