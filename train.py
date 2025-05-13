import os
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
    csv_path = "labels.csv"
    image_dir = "anime"
    checkpoint_dir = "checkpoints"
    samples_dir = "training_samples"
    
    pretrained_model_name = "openai/clip-vit-large-patch14"
    image_size = 256
    batch_size = 1
    num_epochs = 1000
    learning_rate = 1e-4

    device = "cuda" if torch.cuda.is_available() else "cpu"
    samples_per_epoch = 5000

    num_train_timesteps = 500
    beta_start = 0.0001
    beta_end = 0.02

    num_samples = 1
    sample_steps = 50
    sample_seed = 42

# ─── CUDA Setup ────────────────────────────────────────────────────────────────
def setup_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        print(f"CUDA verfügbar - Verwende GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA nicht verfügbar - Verwende CPU")

# ─── Dataset ───────────────────────────────────────────────────────────────────
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
            text, padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt"
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

# ─── Model Initialization ─────────────────────────────────────────────────────
def initialize_models():
    text_encoder = CLIPTextModel.from_pretrained(Config.pretrained_model_name)
    tokenizer = CLIPTokenizer.from_pretrained(Config.pretrained_model_name)
    
    unet = UNet2DConditionModel(
        sample_size=Config.image_size // 8,
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
        num_train_timesteps=Config.num_train_timesteps,
        beta_start=Config.beta_start,
        beta_end=Config.beta_end,
        beta_schedule="linear"
    )
    
    return text_encoder, unet, noise_scheduler, tokenizer

# ─── Checkpoint Handling ───────────────────────────────────────────────────────
def find_latest_checkpoint():
    os.makedirs(Config.checkpoint_dir, exist_ok=True)
    files = [f for f in os.listdir(Config.checkpoint_dir)
             if f.startswith("checkpoint_epoch_") and f.endswith(".pt")]
    if not files:
        return None

    # Liste von (epoch:int, timestamp:datetime, filename)
    ckpts = []
    for fn in files:
        parts = fn.split('_')
        # parts = ['checkpoint','epoch','{E}','YYYYMMDD','HHMMSS.pt']
        epoch = int(parts[2])
        ts_str = parts[3] + "_" + parts[4].replace(".pt","")
        ts = datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
        ckpts.append( (epoch, ts, fn) )

    # Finde höchsten Epoch-Wert
    max_epoch = max(c[0] for c in ckpts)
    # Filter nur die mit max_epoch, und wähle den mit spätestem Timestamp
    latest = max((c for c in ckpts if c[0]==max_epoch), key=lambda x: x[1])
    return os.path.join(Config.checkpoint_dir, latest[2])

def extract_epoch_from_filename(path):
    fn = os.path.basename(path)
    match = re.search(r'checkpoint_epoch_(\d+)_', fn)
    return int(match.group(1)) if match else 0

def save_checkpoint(epoch, unet, optimizer, loss):
    os.makedirs(Config.checkpoint_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(
        Config.checkpoint_dir,
        f"checkpoint_epoch_{epoch}_{timestamp}.pt"
    )
    torch.save({
        'epoch': epoch,
        'unet_state_dict': {k: v.cpu() for k, v in unet.state_dict().items()},
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
    print(f"\nCheckpoint gespeichert: {path}")
    return path

# ─── Sample Generation ────────────────────────────────────────────────────────
def generate_samples(epoch, unet, text_encoder, tokenizer, device):
    os.makedirs(Config.samples_dir, exist_ok=True)
    unet.eval(); text_encoder.eval()
    
    prompts = [
        "a girl with purple hairs",
        "a girl with purple hair",
        "a girl with blue hairs",
        "a woman with purple hairs and a black shirt"
    ]
    ddim = DDIMScheduler(
        num_train_timesteps=Config.num_train_timesteps,
        beta_start=Config.beta_start,
        beta_end=Config.beta_end,
        beta_schedule="linear"
    )
    torch.manual_seed(Config.sample_seed)
    
    with torch.no_grad():
        for i, p in enumerate(prompts[:Config.num_samples]):
            ti = tokenizer(p, padding="max_length",
                           max_length=tokenizer.model_max_length,
                           truncation=True, return_tensors="pt").to(device)
            emb = text_encoder(ti.input_ids)[0]
            noise = torch.randn((1,3,Config.image_size,Config.image_size), device=device)
            ddim.set_timesteps(Config.sample_steps)
            for t in ddim.timesteps:
                pred = unet(noise, t, encoder_hidden_states=emb).sample
                noise = ddim.step(pred, t, noise).prev_sample
            
            out = (noise + 1) / 2
            save_image(out, os.path.join(
                Config.samples_dir,
                f"sample_epoch_{epoch}_{i}.png"
            ))
    unet.train(); text_encoder.train()

def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location='cpu')
    except Exception as e:
        print(f"Fehler beim Laden des Checkpoints: {e}")
        return None

# ─── Training Loop ────────────────────────────────────────────────────────────
def train():
    setup_cuda()
    device = torch.device(Config.device)

    text_encoder, unet, noise_scheduler, tokenizer = initialize_models()
    text_encoder = text_encoder.to(device); unet = unet.to(device)
    text_encoder.train(); unet.train()

    optimizer = optim.AdamW(unet.parameters(), lr=Config.learning_rate)

    # Checkpoint laden
    latest_ckpt = find_latest_checkpoint()
    if latest_ckpt:
        print(f"Versuche letzten Checkpoint zu laden: {latest_ckpt}")
        ckpt = load_checkpoint(latest_ckpt, device)
        if ckpt:
            unet.load_state_dict(ckpt['unet_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            start_epoch = extract_epoch_from_filename(latest_ckpt)
            print(f"Checkpoint geladen, starte mit Epoche {start_epoch}")
        else:
            print("Konnte Checkpoint nicht laden, starte bei Epoche 0")
            start_epoch = 0
    else:
        print("Kein Checkpoint gefunden, starte bei Epoche 0")
        start_epoch = 0

    # Haupt-Loop
    for epoch in range(start_epoch, Config.num_epochs):
        dataloader = prepare_data()
        epoch_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{Config.num_epochs}")
        for batch in pbar:
            optimizer.zero_grad()
            pv = batch["pixel_values"].to(device, non_blocking=True)
            ids = batch["input_ids"].to(device, non_blocking=True)
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
        save_checkpoint(epoch + 1, unet, optimizer, avg)
        generate_samples(epoch + 1, unet, text_encoder, tokenizer, device)

    # Finale Speicherung
    torch.save(unet.state_dict(), "text_to_image_unet_final.pth")
    text_encoder.save_pretrained("text_encoder")
    tokenizer.save_pretrained("tokenizer")
    print("\nTraining abgeschlossen!")

if __name__ == "__main__":
    train()


