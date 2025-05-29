import os
import random
import glob
import re
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

# ─── Konfiguration ─────────────────────────────────────────────────────────────
class Config:
    def __init__(self):
        self.csv_path = input("Pfad zur CSV-Datei (z.B. labels.csv): ") or "labels.csv"
        self.image_dir = input("Bildordner (z.B. img): ") or "img"
        self.checkpoint_dir = input("Checkpoint-Ordner (z.B. checkpoints): ") or "checkpoints"
        self.samples_dir = input("Sample-Ordner (z.B. training_samples): ") or "training_samples"

        self.pretrained_model_name = input("Pretrained Model (z.B. openai/clip-vit-large-patch14): ") or "openai/clip-vit-large-patch14"
        self.image_size = int(input("Bildgröße (z.B. 256): ") or 256)
        self.batch_size = int(input("Batch Size (z.B. 1): ") or 1)
        self.num_epochs = int(input("Anzahl Epochen (z.B. 1000): ") or 1000)
        self.learning_rate = float(input("Lernrate (z.B. 1e-4): ") or 1e-4)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.samples_per_epoch = int(input("Samples pro Epoche (z.B. 10000): ") or 10000)

        self.num_train_timesteps = int(input("Train-Timesteps (z.B. 500): ") or 500)
        self.beta_start = float(input("Beta Start (z.B. 0.0001): ") or 0.0001)
        self.beta_end = float(input("Beta End (z.B. 0.02): ") or 0.02)

        self.num_samples = int(input("Anzahl generierter Samples (z.B. 1): ") or 1)
        self.sample_steps = int(input("Sampling Steps (z.B. 100): ") or 100)
        self.sample_seed = int(input("Sample Seed (z.B. 42): ") or 42)

# ─── CUDA Setup ────────────────────────────────────────────────────────────────
def setup_cuda(config):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        print(f"CUDA verfügbar - Verwende GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA nicht verfügbar - Verwende CPU")
    return torch.device(config.device)

# ─── Dataset ───────────────────────────────────────────────────────────────────
class TextImageDataset(Dataset):
    def __init__(self, config, transform=None):
        self.df = pd.read_csv(config.csv_path)
        self.image_dir = config.image_dir
        self.transform = transform
        self.tokenizer = CLIPTokenizer.from_pretrained(config.pretrained_model_name)
        self.image_size = config.image_size
        
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
            image = Image.new('RGB', (self.image_size, self.image_size))
        
        if self.transform:
            image = self.transform(image)
            
        text_input = self.tokenizer(
            text, padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt"
        )
        
        return {
            "pixel_values": image,
            "input_ids": text_input.input_ids.squeeze()
        }

def create_random_sampler(dataset_size, samples_per_epoch):
    replace = dataset_size < samples_per_epoch
    indices = np.random.choice(dataset_size, samples_per_epoch, replace=replace)
    return SubsetRandomSampler(indices)

def prepare_data(config):
    transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    full_dataset = TextImageDataset(config, transform)
    sampler = create_random_sampler(len(full_dataset), config.samples_per_epoch)
    
    return DataLoader(
        full_dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        drop_last=True,
        pin_memory=True
    )

# ─── Model Initialization ─────────────────────────────────────────────────────
def initialize_models(config):
    text_encoder = CLIPTextModel.from_pretrained(config.pretrained_model_name)
    tokenizer = CLIPTokenizer.from_pretrained(config.pretrained_model_name)
    
    unet = UNet2DConditionModel(
        sample_size=config.image_size // 8,
        in_channels=3, out_channels=3, layers_per_block=2,
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
        num_train_timesteps=config.num_train_timesteps,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
        beta_schedule="linear"
    )
    
    return text_encoder, unet, noise_scheduler, tokenizer

# ─── Checkpoints & Samples ─────────────────────────────────────────────────────
def find_latest_checkpoint(config):
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    files = [f for f in os.listdir(config.checkpoint_dir)
             if f.startswith("checkpoint_epoch_") and f.endswith(".pt")]
    if not files:
        return None

    ckpts = []
    for fn in files:
        parts = fn.split('_')
        epoch = int(parts[2])
        ts_str = parts[3] + "_" + parts[4].replace(".pt", "")
        ts = datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
        ckpts.append((epoch, ts, fn))

    max_epoch = max(c[0] for c in ckpts)
    latest = max((c for c in ckpts if c[0] == max_epoch), key=lambda x: x[1])
    return os.path.join(config.checkpoint_dir, latest[2])

def extract_epoch_from_filename(path):
    fn = os.path.basename(path)
    match = re.search(r'checkpoint_epoch_(\d+)_', fn)
    return int(match.group(1)) if match else 0

def save_checkpoint(config, epoch, unet, optimizer, loss):
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(config.checkpoint_dir, f"checkpoint_epoch_{epoch}_{timestamp}.pt")
    torch.save({
        'epoch': epoch,
        'unet_state_dict': {k: v.cpu() for k, v in unet.state_dict().items()},
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
    print(f"\n✅ Checkpoint gespeichert: {path}")
    return path

def generate_samples(config, epoch, unet, text_encoder, tokenizer, device):
    os.makedirs(config.samples_dir, exist_ok=True)
    unet.eval(); text_encoder.eval()

    prompts = [
        "a girl with purple hairs",
        "a girl with blue hairs",
        "a woman with purple hairs and a black shirt"
    ]

    # ✨ Fixierter Seed für reproduzierbare Ergebnisse
    torch.manual_seed(config.sample_seed)
    np.random.seed(config.sample_seed)

    ddim = DDIMScheduler(
        num_train_timesteps=config.num_train_timesteps,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
        beta_schedule="linear"
    )

    with torch.no_grad():
        for i, p in enumerate(prompts[:config.num_samples]):
            print(f"[Sample {i}] 🎯 Verwendeter Seed: {config.sample_seed}")

            ti = tokenizer(p, padding="max_length",
                           max_length=tokenizer.model_max_length,
                           truncation=True, return_tensors="pt").to(device)
            emb = text_encoder(ti.input_ids)[0]
            noise = torch.randn((1, 3, config.image_size, config.image_size), device=device)
            ddim.set_timesteps(config.sample_steps)

            for t in ddim.timesteps:
                pred = unet(noise, t, encoder_hidden_states=emb).sample
                noise = ddim.step(pred, t, noise).prev_sample

            out = (noise + 1) / 2
            save_image(out, os.path.join(config.samples_dir, f"sample_epoch_{epoch}_{i}.png"))

    unet.train(); text_encoder.train()


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location='cpu')
    except Exception as e:
        print(f"Fehler beim Laden des Checkpoints: {e}")
        return None

# ─── Training ─────────────────────────────────────────────────────────────────
def train():
    config = Config()
    device = setup_cuda(config)

    text_encoder, unet, noise_scheduler, tokenizer = initialize_models(config)
    text_encoder = text_encoder.to(device); unet = unet.to(device)
    text_encoder.train(); unet.train()

    optimizer = optim.AdamW(unet.parameters(), lr=config.learning_rate)

    latest_ckpt = find_latest_checkpoint(config)
    if latest_ckpt:
        print(f"🧠 Lade Checkpoint: {latest_ckpt}")
        ckpt = load_checkpoint(latest_ckpt, device)
        if ckpt:
            unet.load_state_dict(ckpt['unet_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            start_epoch = extract_epoch_from_filename(latest_ckpt)
        else:
            start_epoch = 0
    else:
        start_epoch = 0

    for epoch in range(start_epoch, config.num_epochs):
        dataloader = prepare_data(config)
        epoch_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        for batch in pbar:
            optimizer.zero_grad()
            pv = batch["pixel_values"].to(device)
            ids = batch["input_ids"].to(device)
            with torch.no_grad():
                emb = text_encoder(ids)[0]
            noise = torch.randn_like(pv)
            ts = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                               (pv.shape[0],), device=device).long()
            noisy = noise_scheduler.add_noise(pv, noise, ts)
            pred = unet(noisy, ts, encoder_hidden_states=emb).sample
            loss = nn.functional.mse_loss(pred, noise)
            loss.backward(); optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg = epoch_loss / len(dataloader)
        save_checkpoint(config, epoch + 1, unet, optimizer, avg)
        generate_samples(config, epoch + 1, unet, text_encoder, tokenizer, device)

    torch.save(unet.state_dict(), "text_to_image_unet_final.pth")
    text_encoder.save_pretrained("text_encoder")
    tokenizer.save_pretrained("tokenizer")
    print("\n🚀 Training abgeschlossen!")

if __name__ == "__main__":
    train()


