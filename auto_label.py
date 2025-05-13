import os
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import csv

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

def auto_label_images(image_folder="anime", output_file="labels.csv", batch_size=8):
    os.makedirs(image_folder, exist_ok=True)
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    
    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image_name", "caption"])
        
        for i in range(0, len(image_files), batch_size):
            batch_files = image_files[i:i + batch_size]
            images = []
            
            for img_name in batch_files:
                img_path = os.path.join(image_folder, img_name)
                image = Image.open(img_path).convert("RGB")
                images.append(image)
            
            # Batch-Verarbeitung
            inputs = processor(images, return_tensors="pt", padding=True).to(device)
            outputs = model.generate(**inputs, max_new_tokens=50)
            captions = processor.batch_decode(outputs, skip_special_tokens=True)
            
            for img_name, caption in zip(batch_files, captions):
                writer.writerow([img_name, caption])
                print(f"Gelabelt: {img_name} -> {caption}")

if __name__ == "__main__":
    auto_label_images(batch_size=265)  # Batch-Größe anpassbar (abhängig von GPU-Speicher)
