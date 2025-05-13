import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from transformers import AutoTokenizer
import numpy as np

class TextImageDataset(Dataset):
    def __init__(self, vae, image_folder="anime", csv_file="labels.csv", resolution=1024):
        self.image_folder = image_folder
        self.data = pd.read_csv(csv_file)
        self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.resolution = resolution
        
        # Precompute latents with proper type handling
        self.latents = []
        for idx in range(len(self.data)):
            img_name = self.data.iloc[idx, 0]
            img_path = os.path.join(self.image_folder, img_name)
            
            # Load and preprocess image
            image = Image.open(img_path).convert("RGB")
            image = image.resize((resolution, resolution))
            image = np.array(image)
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # Ensure float32
            image = (image / 127.5 - 1.0).unsqueeze(0).to(vae.device)
            
            # Encode to latent space
            with torch.no_grad():
                latent = vae.encode(image).latent_dist.sample() * 0.18215
                self.latents.append(latent.cpu())
                
        print(f"✅ {len(self.latents)} images precomputed")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        caption = self.data.iloc[idx, 1]
        inputs = self.tokenizer(
            caption, 
            max_length=self.tokenizer.model_max_length, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )
        return self.latents[idx].squeeze(0), inputs.input_ids.squeeze(0)
