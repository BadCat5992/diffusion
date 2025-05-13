import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from diffusers import UNet2DConditionModel, DDPMScheduler, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm
from datetime import datetime

# Konfiguration
class Config:
    # Dateipfade
    csv_path = "labels.csv"
    image_dir = "anime"
    checkpoint_dir = "checkpoints"
    samples_dir = "training_samples"
    
    # Modellparameter
    pretrained_model_name = "openai/clip-vit-large-patch14"
    image_size = 256
    batch_size = 1
    num_epochs = 1000
    learning_rate = 1e-4

    device = "cuda" if torch.cuda.is_available() else "cpu"
    samples_per_epoch = 5000

    # Diffusion Parameter
    num_train_timesteps = 500
    beta_start = 0.0001
    beta_end = 0.02

    # Sampling Parameter
    num_samples = 1
    sample_steps = 50
    sample_seed = 42

# CUDA Initialisierung
def setup_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        print(f"CUDA verfügbar - Verwende GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA nicht verfügbar - Verwende CPU")

# Dataset Klasse
class TextImageDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform
        self.tokenizer = CLIPTokenizer.from_pretrained(Config.pretrained_model_name)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['image_name']
        text = row['caption']
        
        img_path = os.path.join(self.image_dir, img_name)
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            image = Image.new('RGB', (Config.image_size, Config.image_size))
        
        if self.transform:
            image = self.transform(image)
            
        text_input = self.tokenizer(
            text, 
            padding="max_length", 
            max_length=self.tokenizer.model_max_length, 
            truncation=True, 
            return_tensors="pt"
        )
        
        return {
            "pixel_values": image,
            "input_ids": text_input.input_ids.squeeze()
        }

def create_random_sampler(dataset_size):
    replace = dataset_size < Config.samples_per_epoch
    indices = np.random.choice(dataset_size, Config.samples_per_epoch, replace=replace)
    return SubsetRandomSampler(indices)

def prepare_data():
    transform = transforms.Compose([
        transforms.Resize((Config.image_size, Config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    full_dataset = TextImageDataset(Config.csv_path, Config.image_dir, transform)
    sampler = create_random_sampler(len(full_dataset))
    
    return DataLoader(
        full_dataset,
        batch_size=Config.batch_size,
        sampler=sampler,
        drop_last=True,
        pin_memory=True
    )

def initialize_models():
    text_encoder = CLIPTextModel.from_pretrained(Config.pretrained_model_name)
    tokenizer = CLIPTokenizer.from_pretrained(Config.pretrained_model_name)
    
    unet = UNet2DConditionModel(
        sample_size=Config.image_size // 8,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 256, 512, 512),
        down_block_types=(
            "DownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "UpBlock2D",
        ),
        cross_attention_dim=text_encoder.config.hidden_size,
    )
    
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=Config.num_train_timesteps,
        beta_start=Config.beta_start,
        beta_end=Config.beta_end,
        beta_schedule="linear"
    )
    
    return text_encoder, unet, noise_scheduler, tokenizer

def find_latest_checkpoint():
    os.makedirs(Config.checkpoint_dir, exist_ok=True)
    checkpoints = glob.glob(os.path.join(Config.checkpoint_dir, "checkpoint_epoch_*.pt"))
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: int(x.split('_')[-2]))
    return checkpoints[-1]

def save_checkpoint(epoch, unet, optimizer, loss):
    os.makedirs(Config.checkpoint_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_path = os.path.join(
        Config.checkpoint_dir, 
        f"checkpoint_epoch_{epoch+1}_{timestamp}.pt"
    )
    torch.save({
        'epoch': epoch,
        'unet_state_dict': {k: v.cpu() for k, v in unet.state_dict().items()},
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    print(f"\nCheckpoint gespeichert: {checkpoint_path}")
    return checkpoint_path

def generate_samples(epoch, unet, text_encoder, tokenizer, device):
    os.makedirs(Config.samples_dir, exist_ok=True)
    unet.eval()
    text_encoder.eval()
    
    sample_texts = [
        "a girl with purple hairs",
        "a girl with purple hair",
        "a girl with blue hairs",
        "a woman with purple hairs and a black shirt"
    ]
    
    ddim_scheduler = DDIMScheduler(
        num_train_timesteps=Config.num_train_timesteps,
        beta_start=Config.beta_start,
        beta_end=Config.beta_end,
        beta_schedule="linear"
    )
    
    torch.manual_seed(Config.sample_seed)
    
    with torch.no_grad():
        for i, prompt in enumerate(sample_texts[:Config.num_samples]):
            text_input = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            ).to(device)
            
            text_embeddings = text_encoder(text_input.input_ids)[0]
            noise = torch.randn((1, 3, Config.image_size, Config.image_size), device=device)
            ddim_scheduler.set_timesteps(Config.sample_steps)
            
            for t in ddim_scheduler.timesteps:
                noise_pred = unet(
                    noise,
                    t,
                    encoder_hidden_states=text_embeddings
                ).sample
                noise = ddim_scheduler.step(noise_pred, t, noise).prev_sample
            
            sample_path = os.path.join(
                Config.samples_dir,
                f"sample_epoch_{epoch+1}_{i}_{prompt[:20].replace(' ', '_')}.png"
            )
            save_image((noise + 1) / 2, sample_path)
            print(f"Sample gespeichert: {sample_path}")
    
    unet.train()
    text_encoder.train()

def load_checkpoint(checkpoint_path, device):
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        # Tensoren verschieben wir im train(), deshalb hier nur Rückgabe
        return checkpoint
    except Exception as e:
        print(f"Fehler beim Laden des Checkpoints: {e}")
        return None

def train():
    setup_cuda()
    device = torch.device(Config.device)
    
    # Modelle initialisieren
    text_encoder, unet, noise_scheduler, tokenizer = initialize_models()
    
    # Modelle auf Device verschieben
    text_encoder = text_encoder.to(device)
    unet = unet.to(device)
    text_encoder.train()
    unet.train()

    # Optimizer initialisieren (erst nachdem Unet auf GPU ist)
    optimizer = optim.AdamW(unet.parameters(), lr=Config.learning_rate)
    
    # Checkpoint laden falls vorhanden
    start_epoch = 0
    latest_ckpt = find_latest_checkpoint()
    if latest_ckpt:
        print(f"Versuche letzten Checkpoint zu laden: {latest_ckpt}")
        ckpt = load_checkpoint(latest_ckpt, device)
        if ckpt:
            unet.load_state_dict(ckpt['unet_state_dict'])
            start_epoch = ckpt['epoch'] + 1
            print(f"Checkpoint erfolgreich geladen, starte mit Epoche {start_epoch}")
            if 'optimizer_state_dict' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        else:
            print("Checkpoint konnte nicht geladen werden, starte von vorne")
    
    # Trainingsloop
    for epoch in range(start_epoch, Config.num_epochs):
        dataloader = prepare_data()
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{Config.num_epochs}")
        
        for batch in progress_bar:
            optimizer.zero_grad()
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            
            with torch.no_grad():
                encoder_hidden_states = text_encoder(input_ids)[0]
            
            noise = torch.randn_like(pixel_values)
            timesteps = torch.randint(
                0, noise_scheduler.num_train_timesteps,
                (pixel_values.shape[0],),
                device=device
            ).long()
            
            noisy_images = noise_scheduler.add_noise(pixel_values, noise, timesteps)
            noise_pred = unet(noisy_images, timesteps, encoder_hidden_states).sample
            loss = nn.functional.mse_loss(noise_pred, noise)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        
        avg_loss = epoch_loss / len(dataloader)
        save_checkpoint(epoch, unet, optimizer, avg_loss)
        generate_samples(epoch, unet, text_encoder, tokenizer, device)
    
    # Finale Speicherung
    final_model_path = "text_to_image_unet_final.pth"
    torch.save(unet.state_dict(), final_model_path)
    text_encoder.save_pretrained("text_encoder")
    tokenizer.save_pretrained("tokenizer")
    print(f"\nTraining abgeschlossen. Finales Modell gespeichert unter {final_model_path}")

if __name__ == "__main__":
    train()

