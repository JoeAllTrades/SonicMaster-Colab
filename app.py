import os
import sys
import yaml
import torch
import torchaudio
import soundfile as sf
import numpy as np
import gradio as gr
from pathlib import Path
from safetensors.torch import load_file
from diffusers import AutoencoderOobleck

# Import local modules
from model import TangoFlux

# Setup paths
SPACE_ROOT = Path(__file__).parent.resolve()
sys.path.append(SPACE_ROOT.as_posix())
CONFIG_PATH = SPACE_ROOT / "configs" / "tangoflux_config.yaml"
WEIGHTS_DIR = SPACE_ROOT / "weights"

# Global model state
MODEL = None
VAE = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_models(hf_token=None):
    """Loads the Model and VAE into memory once."""
    global MODEL, VAE
    
    if MODEL is not None and VAE is not None:
        return "Models already loaded."

    print("⏳ Loading configuration...")
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    print("⏳ Loading SonicMaster weights...")
    # Ensure weights exist (downloaded via snapshot in Colab)
    # Checks standard HF cache or local weights folder
    model_path = SPACE_ROOT / "SonicMaster" / "model.safetensors" # check subdir
    if not model_path.exists():
        # Fallback to current dir or weights dir
        model_path = SPACE_ROOT / "model.safetensors"
    
    if not model_path.exists():
         # Last resort: try standard HF download location logic or let user know
         # For Colab usage, we assume snapshot_download put it in root or we download it now
         from huggingface_hub import hf_hub_download
         model_path = hf_hub_download(repo_id="amaai-lab/SonicMaster", filename="model.safetensors")

    # Load TangoFlux
    model = TangoFlux(config=config["model"])
    weights = load_file(str(model_path))
    model.load_state_dict(weights, strict=False)
    model.to(DEVICE).eval()
    
    # Freeze text encoder
    for param in model.text_encoder.parameters():
        param.requires_grad = False
    
    MODEL = model

    print("⏳ Loading VAE (Stable Audio Open 1.0)...")
    try:
        vae = AutoencoderOobleck.from_pretrained(
            "stabilityai/stable-audio-open-1.0", 
            subfolder="vae",
            token=hf_token
        ).to(DEVICE)
        vae.eval()
        VAE = vae
    except Exception as e:
        return f"Error loading VAE. Check HF Token! {e}"

    return "✅ Models loaded successfully!"

def process_audio(
    input_audio, 
    prompt, 
    steps, 
    guidance_scale, 
    seed, 
    full_song_mode,
    overlap_duration=10
):
    if MODEL is None or VAE is None:
        raise gr.Error("Models not loaded! Please enter HF Token and load models first.")

    if not input_audio:
        raise gr.Error("Please upload audio.")

    # Prepare inputs
    torch.manual_seed(seed)
    prompt = prompt if prompt.strip() else "Master this track"
    
    # Load Audio
    sr, audio_data = input_audio
    # Convert to torch tensor [C, T]
    if audio_data.dtype == np.int16:
        audio_data = audio_data / 32768.0
    elif audio_data.dtype == np.int32:
        audio_data = audio_data / 2147483648.0
        
    audio_tensor = torch.from_numpy(audio_data).float()
    
    # Handle Mono/Stereo
    if audio_tensor.ndim == 1:
        audio_tensor = audio_tensor.unsqueeze(0).repeat(2, 1)
    elif audio_tensor.ndim == 2:
        audio_tensor = audio_tensor.t() # [T, C] -> [C, T]
        if audio_tensor.shape[0] == 1:
            audio_tensor = audio_tensor.repeat(2, 1)
    
    # Resample to 44100
    target_fs = 44100
    if sr != target_fs:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_fs)
        audio_tensor = resampler(audio_tensor)

    audio_tensor = audio_tensor.to(DEVICE)
    
    # Chunking Parameters
    chunk_duration = 30 # Model trained on 30s
    chunk_size = chunk_duration * target_fs
    overlap_size = overlap_duration * target_fs
    
    if not full_song_mode:
        # Simple mode: Crop or Pad to 30s
        if audio_tensor.shape[1] > chunk_size:
            audio_tensor = audio_tensor[:, :chunk_size]
        elif audio_tensor.shape[1] < chunk_size:
            pad_len = chunk_size - audio_tensor.shape[1]
            audio_tensor = torch.nn.functional.pad(audio_tensor, (0, pad_len))
        
        chunks = [audio_tensor]
        stride = chunk_size
    else:
        # Full Song Mode: Split with overlap
        chunks = []
        start = 0
        stride = chunk_size - overlap_size
        while start < audio_tensor.shape[1]:
            end = min(start + chunk_size, audio_tensor.shape[1])
            chunk = audio_tensor[:, start:end]
            # Pad last chunk
            if chunk.shape[1] < chunk_size:
                chunk = torch.nn.functional.pad(chunk, (0, chunk_size - chunk.shape[1]))
            chunks.append(chunk)
            start += stride

    # Processing Loop
    print(f"Processing {len(chunks)} chunks...")
    
    # Encode all chunks first (Batched)
    degraded_latents_list = []
    batch_size_vae = 4 
    
    with torch.no_grad():
        chunk_stack = torch.stack(chunks).to(DEVICE)
        for b in range(0, len(chunk_stack), batch_size_vae):
            batch = chunk_stack[b : b + batch_size_vae]
            lat = VAE.encode(batch).latent_dist.mode()
            degraded_latents_list.append(lat)
        degraded_latents = torch.cat(degraded_latents_list, dim=0) # [N, C, T']

    decoded_waves = []
    prev_cond_latent = None # For full song continuity

    # Inference Loop
    for i in range(len(degraded_latents)):
        latent_in = degraded_latents[i].unsqueeze(0).transpose(1, 2) # [1, T, C]
        
        # Determine conditioning (First chunk vs subsequent)
        cond = prev_cond_latent if full_song_mode and i > 0 else None
        
        with torch.no_grad():
            result_latent = MODEL.inference_flow(
                latent_in,
                prompt,
                audiocond_latents=cond,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                duration=chunk_duration,
                seed=seed,
                disable_progress=False,
                num_samples_per_prompt=1,
                solver="Euler"
            )
            
            # Decode
            decoded = VAE.decode(result_latent.transpose(2, 1)).sample.cpu() # [1, 2, T]
            decoded_waves.append(decoded)
            
            # Prepare condition for next chunk (Last seconds of current output)
            if full_song_mode:
                last_segment = decoded[:, :, -overlap_size:].to(DEVICE)
                prev_cond_latent = VAE.encode(last_segment).latent_dist.mode().transpose(1, 2)

    # Stitching / Reconstruction
    if not full_song_mode:
        final_audio = decoded_waves[0].squeeze(0)
    else:
        # Crossfade stitching
        final_audio = decoded_waves[0] # [1, 2, T]
        for i in range(1, len(decoded_waves)):
            prev = final_audio[:, :, -overlap_size:]
            curr = decoded_waves[i][:, :, :overlap_size]
            
            # Linear crossfade
            alpha = torch.linspace(1, 0, steps=overlap_size).view(1, 1, -1)
            beta = 1 - alpha
            blended = prev * alpha + curr * beta
            
            final_audio = torch.cat([
                final_audio[:, :, :-overlap_size],
                blended,
                decoded_waves[i][:, :, overlap_size:]
            ], dim=2)
        
        final_audio = final_audio.squeeze(0)

    # Export
    output_path = SPACE_ROOT / "output.flac"
    sf.write(output_path, final_audio.numpy().T, target_fs)
    
    return output_path.as_posix()


# --- UI Construction ---
with gr.Blocks(title="SonicMaster Colab", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🎛️ SonicMaster: AI Music Restoration")
    
    with gr.Row():
        with gr.Column(variant="panel"):
            gr.Markdown("### 1. Setup")
            hf_token_inp = gr.Textbox(label="Hugging Face Token (Required)", type="password", placeholder="hf_...")
            load_btn = gr.Button("🔌 Load Model", variant="primary")
            load_status = gr.Textbox(label="Status", interactive=False, value="Waiting to load...")
            
        with gr.Column(variant="panel"):
            gr.Markdown("### 2. Input")
            input_audio = gr.Audio(label="Source Audio", type="numpy")
            prompt = gr.Textbox(label="Prompt", value="Master this track, remove reverb, high fidelity", lines=2)
            
            with gr.Accordion("⚙️ Advanced Settings", open=True):
                full_song_chk = gr.Checkbox(label="Full Song Mode (Stitch Chunks)", value=True)
                steps = gr.Slider(10, 100, value=25, step=1, label="Inference Steps")
                cfg = gr.Slider(1, 15, value=7.0, step=0.5, label="Guidance Scale")
                seed = gr.Number(label="Seed", value=42)
            
            run_btn = gr.Button("🚀 Process Audio", variant="primary", interactive=False)

    with gr.Row():
         output_audio = gr.Audio(label="Restored Audio", type="filepath")

    # Events
    def unlock_ui(status):
        if "successfully" in status:
            return gr.update(interactive=True)
        return gr.update(interactive=False)

    load_btn.click(load_models, inputs=[hf_token_inp], outputs=[load_status]).then(
        unlock_ui, inputs=[load_status], outputs=[run_btn]
    )
    
    run_btn.click(
        process_audio,
        inputs=[input_audio, prompt, steps, cfg, seed, full_song_chk],
        outputs=[output_audio]
    )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)