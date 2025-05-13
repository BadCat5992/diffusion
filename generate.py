import torch
from torchvision import transforms
from PIL import Image
import gradio as gr
from diffusers import UNet2DConditionModel, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
import os
import glob
from datetime import datetime

# Konfiguration (wie in deinem Training)
class Config:
    pretrained_model_name = "openai/clip-vit-large-patch14"
    image_size = 256
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sample_steps = 50
    num_train_timesteps = 500
    beta_start = 0.0001
    beta_end = 0.02
    checkpoint_dir = "checkpoints"  # Verzeichnis mit den Checkpoints

# Hilfsfunktion zum Finden des neuesten Checkpoints
def find_latest_checkpoint():
    os.makedirs(Config.checkpoint_dir, exist_ok=True)
    checkpoints = glob.glob(os.path.join(Config.checkpoint_dir, "checkpoint_epoch_*.pt"))
    if not checkpoints:
        return None
    
    # Extrahiere Epochennummer und Zeitstempel aus dem Dateinamen
    def get_sort_key(filename):
        parts = filename.split('_')
        epoch = int(parts[-3])  # Die drittletzte Zahl ist die Epochennummer
        date_part = parts[-2]   # Datumsteil (YYYYMMDD)
        time_part = parts[-1].split('.')[0]  # Zeitteil (HHMMSS)
        return (epoch, date_part + time_part)  # Sortiere nach Epoche, dann Zeitstempel
    
    # Sortiere absteigend nach Epoche und Zeitstempel
    checkpoints.sort(key=get_sort_key, reverse=True)
    return checkpoints[0]

# Modell initialisieren
def load_models():
    # Text Encoder und Tokenizer laden
    text_encoder = CLIPTextModel.from_pretrained(Config.pretrained_model_name)
    tokenizer = CLIPTokenizer.from_pretrained(Config.pretrained_model_name)
    
    # UNet initialisieren (wie im Training)
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
    
    # Scheduler initialisieren
    ddim_scheduler = DDIMScheduler(
        num_train_timesteps=Config.num_train_timesteps,
        beta_start=Config.beta_start,
        beta_end=Config.beta_end,
        beta_schedule="linear"
    )
    
    # Gerät auswählen
    device = torch.device(Config.device)
    text_encoder = text_encoder.to(device)
    unet = unet.to(device)
    
    return text_encoder, tokenizer, unet, ddim_scheduler, device

# Checkpoint laden
def load_checkpoint(unet, checkpoint_path):
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Zustand laden und auf das richtige Gerät verschieben
        unet.load_state_dict({k: v.to(Config.device) for k, v in checkpoint['unet_state_dict'].items()})
        
        # Informationen über den geladenen Checkpoint ausgeben
        epoch = checkpoint.get('epoch', 'unbekannt')
        loss = checkpoint.get('loss', 'unbekannt')
        print(f"Checkpoint erfolgreich geladen (Epoche {epoch}, Loss: {loss:.4f}) von {checkpoint_path}")
        return True
    except Exception as e:
        print(f"Fehler beim Laden des Checkpoints: {e}")
        return False

# Bildgenerierung
def generate_image(prompt, num_steps, seed):
    # Modelle in den Evaluationsmodus versetzen
    text_encoder.eval()
    unet.eval()
    
    # Seed setzen für Reproduzierbarkeit
    torch.manual_seed(seed)
    
    try:
        # Text kodieren
        with torch.no_grad():
            text_input = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            ).to(device)
            
            text_embeddings = text_encoder(text_input.input_ids)[0]
            
            # Rauschen generieren
            noise = torch.randn((1, 3, Config.image_size, Config.image_size), device=device)
            
            # Sampling-Schritte setzen
            ddim_scheduler.set_timesteps(num_steps)
            
            # Fortschrittsbalken für Gradio
            progress = []
            total_steps = len(ddim_scheduler.timesteps)
            
            # Denoising-Prozess
            for i, t in enumerate(ddim_scheduler.timesteps):
                noise_pred = unet(
                    noise,
                    t,
                    encoder_hidden_states=text_embeddings
                ).sample
                noise = ddim_scheduler.step(noise_pred, t, noise).prev_sample
                
                # Fortschritt speichern (für Gradio)
                progress.append((i + 1) / total_steps)
                yield {"progress": progress}, None
            
            # Bild normalisieren und zurückgeben
            image = (noise + 1) / 2
            image = image.clamp(0, 1)
            image = image.cpu().detach().squeeze().permute(1, 2, 0).numpy()
            
            yield {"progress": [1.0]}, image
            
    except Exception as e:
        print(f"Fehler bei der Generierung: {e}")
        yield {"progress": [0.0]}, None

# Gradio Interface mit Fortschrittsbalken
def create_interface():
    with gr.Blocks(title="Anime Diffusion Generator") as demo:
        gr.Markdown("""
        # 🎨 Anime Bildgenerator
        Generiere Anime-Bilder mit deinem trainierten Diffusionsmodell
        """)
        
        with gr.Row():
            with gr.Column():
                prompt_input = gr.Textbox(
                    label="Beschreibung",
                    placeholder="z.B. 'a girl with purple hair and blue eyes'",
                    lines=3,
                    max_lines=5
                )
                with gr.Accordion("Erweiterte Einstellungen", open=False):
                    steps_slider = gr.Slider(
                        minimum=10, maximum=100, value=50, step=5,
                        label="Anzahl der Denoising-Schritte"
                    )
                    seed_input = gr.Number(
                        value=42, label="Seed für Zufallsgenerator"
                    )
                generate_btn = gr.Button("Bild generieren", variant="primary")
                
                # Fortschrittsbalken
                progress_bar = gr.Slider(
                    label="Fortschritt",
                    minimum=0, maximum=1, value=0,
                    interactive=False
                )
            
            with gr.Column():
                output_image = gr.Image(
                    label="Generiertes Bild",
                    elem_id="output-image"
                )
        
        # Beispiele
        examples = gr.Examples(
            examples=[
                ["a beautiful anime girl with long purple hair and green eyes", 50, 42],
                ["a cute anime boy with short blue hair and glasses", 50, 123],
                ["a cool anime character with red eyes and black coat", 50, 456],
                ["anime landscape with cherry blossoms and mountains", 60, 789]
            ],
            inputs=[prompt_input, steps_slider, seed_input],
            label="Beispiel-Prompts"
        )
        
        # Event-Handler
        generate_btn.click(
            fn=generate_image,
            inputs=[prompt_input, steps_slider, seed_input],
            outputs=[progress_bar, output_image],
            show_progress="hidden"
        )
    
    return demo

if __name__ == "__main__":
    # Modelle laden
    print("Initialisiere Modelle...")
    text_encoder, tokenizer, unet, ddim_scheduler, device = load_models()
    
    # Neuesten Checkpoint finden und laden
    print("Suche nach dem neuesten Checkpoint...")
    latest_checkpoint = find_latest_checkpoint()
    
    if latest_checkpoint:
        print(f"Lade Checkpoint: {latest_checkpoint}")
        load_checkpoint(unet, latest_checkpoint)
    else:
        print("Warnung: Kein Checkpoint gefunden! Es wird ein zufällig initialisiertes Modell verwendet.")
    
    # Gradio Interface erstellen und starten
    print("Starte Gradio Interface...")
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,  # Auf True setzen für öffentliche URL
        favicon_path=None
        
    )
