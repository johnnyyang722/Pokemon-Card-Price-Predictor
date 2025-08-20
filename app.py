import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import requests
import re

# ===== CONFIG =====
MODEL_PATH = "pokemon_price_predictor_augmented_finetuned.pt"
# Updated Google Drive file ID
MODEL_URL = "https://drive.google.com/uc?export=download&id=1-H2_KQ75XXARuKfYEgMjsct5h3PTOv0M"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== HELPER FUNCTION TO DOWNLOAD MODEL =====
def download_model(url, save_path):
    st.info("Downloading model, please wait...")
    
    # Transform Google Drive URL to direct download
    file_id = re.search(r"id=([a-zA-Z0-9_-]+)", url).group(1)
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"

    response = requests.get(download_url, stream=True)
    with open(save_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    st.success("Model downloaded!")

# ===== DEFINE MODEL =====
class PricePredictor(nn.Module):
    def __init__(self, rarity_size):
        super().__init__()
        from torchvision import models
        self.cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.cnn.fc = nn.Identity()
        cnn_out = 512
        self.fc = nn.Sequential(
            nn.Linear(cnn_out + rarity_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, image, rarity):
        img_feat = self.cnn(image)
        combined = torch.cat([img_feat, rarity], dim=1)
        return self.fc(combined)

# ===== CHECK MODEL =====
if not os.path.exists(MODEL_PATH):
    download_model(MODEL_URL, MODEL_PATH)

# ===== LOAD MODEL SAFELY =====
RARITY_LIST = ["Common", "Uncommon", "Rare", "Ultra Rare", "Secret Rare"]
model = PricePredictor(rarity_size=len(RARITY_LIST)).to(DEVICE)

try:
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint)
    model.eval()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")

# ===== STREAMLIT APP =====
st.title("Pokémon Card Price Predictor")

card_name = st.text_input("Enter card name:")
rarity = st.selectbox("Select card rarity:", RARITY_LIST)
uploaded_file = st.file_uploader("Upload card image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Card", use_column_width=True)

    # ===== IMAGE PREPROCESSING =====
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # ===== RARITY TO TENSOR =====
    rarity_tensor = torch.zeros(1, len(RARITY_LIST)).to(DEVICE)
    rarity_idx = RARITY_LIST.index(rarity)
    rarity_tensor[0, rarity_idx] = 1.0

    # ===== PREDICTION =====
    with torch.no_grad():
        log_pred = model(image_tensor, rarity_tensor).item()
        price_pred = np.expm1(log_pred)  # reverse log transform

    st.success(f"Predicted Price: ${price_pred:.2f}")
    st.write(f"Card Name: {card_name}")
    st.write(f"Rarity: {rarity}")








