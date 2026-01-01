import os
import sys
import yaml
import torch
import random
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

# Global model state
MODEL = None
VAE = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_models(hf_token=None):
    global MODEL, VAE
    if MODEL is not None and VAE is not None:
        return "✅ Models already loaded."

    print("⏳ Loading configuration...")
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    print("⏳ Searching for SonicMaster weights...")
    possible_paths = [
        SPACE_ROOT / "model.safetensors",
        SPACE_ROOT / "SonicMaster" / "model.safetensors",
        SPACE_ROOT / "weights" / "model.safetensors"
    ]
    model_path = next((p for p in possible_paths if p.exists()), None)
            
    if not model_path:
         print("⬇️ Weights not found locally. Downloading...")
         from huggingface_hub import hf_hub_download
         model_path = hf_hub_download(repo_id="amaai-lab/SonicMaster", filename="model.safetensors")

    print(f"⏳ Loading Main Model from {model_path}...")
    model = TangoFlux(config=config["model"])
    weights = load_file(str(model_path))
    model.load_state_dict(weights, strict=False)
    model.to(DEVICE).eval()
    
    for param in model.text_encoder.parameters():
        param.requires_grad = False
    MODEL = model

    print("⏳ Loading VAE...")
    try:
        vae = AutoencoderOobleck.from_pretrained(
            "stabilityai/stable-audio-open-1.0", 
            subfolder="vae",
            token=hf_token
        ).to(DEVICE)
        vae.eval()
        VAE = vae
    except Exception as e:
        return f"❌ Error loading VAE: {e}"

    return "✅ Models loaded successfully!"

def match_target_amplitude(sound, target_sound, epsilon=1e-4):
    def rms(x): return np.sqrt(np.mean(x**2))
    valid_mask = ~np.isnan(sound) & ~np.isinf(sound)
    sound = np.where(valid_mask, sound, 0)
    source_rms = rms(sound)
    target_rms = rms(target_sound)
    if source_rms < epsilon: return sound
    gain = target_rms / (source_rms + epsilon)
    return sound * min(gain, 5.0)

def process_audio(
    input_audio, 
    prompt, 
    steps, 
    guidance_scale, 
    seed, 
    solver, # <-- New Argument
    full_song_mode, 
    normalize_output, 
    overlap_duration=10
):
    if MODEL is None or VAE is None:
        raise gr.Error("⚠️ Models not loaded!")
    if not input_audio:
        raise gr.Error("⚠️ Please upload audio.")

    # --- Strict Seeding ---
    if seed == -1: seed = random.randint(0, 2**32 - 1)
    print(f"🎲 Seed: {seed} | Solver: {solver}")
    
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    prompt = prompt if prompt.strip() else "Master this track"
    sr, audio_data = input_audio
    
    # Normalize input
    if audio_data.dtype == np.int16: audio_data = audio_data / 32768.0
    elif audio_data.dtype == np.int32: audio_data = audio_data / 2147483648.0
    audio_tensor = torch.from_numpy(audio_data).float()
    original_audio_numpy = audio_tensor.numpy().flatten()

    # Stereo fix
    if audio_tensor.ndim == 1: audio_tensor = audio_tensor.unsqueeze(0).repeat(2, 1)
    elif audio_tensor.ndim == 2:
        audio_tensor = audio_tensor.t()
        if audio_tensor.shape[0] == 1: audio_tensor = audio_tensor.repeat(2, 1)
    
    # Resample
    target_fs = 44100
    if sr != target_fs:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_fs)
        audio_tensor = resampler(audio_tensor)
    audio_tensor = audio_tensor.to(DEVICE)
    
    # Chunking
    chunk_duration = 30
    chunk_size = chunk_duration * target_fs
    overlap_size = overlap_duration * target_fs
    
    if not full_song_mode:
        if audio_tensor.shape[1] > chunk_size:
            audio_tensor = audio_tensor[:, :chunk_size]
        elif audio_tensor.shape[1] < chunk_size:
            audio_tensor = torch.nn.functional.pad(audio_tensor, (0, chunk_size - audio_tensor.shape[1]))
        chunks = [audio_tensor]
    else:
        chunks = []
        start = 0
        stride = chunk_size - overlap_size
        while start < audio_tensor.shape[1]:
            end = min(start + chunk_size, audio_tensor.shape[1])
            chunk = audio_tensor[:, start:end]
            if chunk.shape[1] < chunk_size:
                chunk = torch.nn.functional.pad(chunk, (0, chunk_size - chunk.shape[1]))
            chunks.append(chunk)
            start += stride

    print(f"🔄 Processing {len(chunks)} chunk(s)...")
    
    # Encode
    degraded_latents_list = []
    with torch.no_grad():
        chunk_stack = torch.stack(chunks).to(DEVICE)
        for b in range(0, len(chunk_stack), 4):
            batch = chunk_stack[b : b + 4]
            lat = VAE.encode(batch).latent_dist.mode()
            degraded_latents_list.append(lat)
        degraded_latents = torch.cat(degraded_latents_list, dim=0)

    decoded_waves = []
    prev_cond_latent = None

    # Inference Loop
    for i in range(len(degraded_latents)):
        torch.manual_seed(seed) # Reset seed for consistency per chunk
        
        latent_in = degraded_latents[i].unsqueeze(0).transpose(1, 2)
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
                disable_progress=True,
                num_samples_per_prompt=1,
                solver=solver # Pass solver here
            )
            decoded = VAE.decode(result_latent.transpose(2, 1)).sample.cpu()
            decoded_waves.append(decoded)
            
            if full_song_mode:
                last_segment = decoded[:, :, -overlap_size:].to(DEVICE)
                prev_cond_latent = VAE.encode(last_segment).latent_dist.mode().transpose(1, 2)
        
        torch.cuda.empty_cache()

    # Stitching
    if not full_song_mode:
        final_audio = decoded_waves[0].squeeze(0)
    else:
        final_audio = decoded_waves[0]
        for i in range(1, len(decoded_waves)):
            prev = final_audio[:, :, -overlap_size:]
            curr = decoded_waves[i][:, :, :overlap_size]
            alpha = torch.linspace(1, 0, steps=overlap_size).view(1, 1, -1)
            beta = 1 - alpha
            blended = prev * alpha + curr * beta
            final_audio = torch.cat([
                final_audio[:, :, :-overlap_size],
                blended,
                decoded_waves[i][:, :, overlap_size:]
            ], dim=2)
        final_audio = final_audio.squeeze(0)

    # Normalize
    final_numpy = final_audio.numpy().T
    if normalize_output:
        print("🔊 Normalizing...")
        if final_numpy.ndim > 1:
            for ch in range(final_numpy.shape[1]):
                final_numpy[:, ch] = match_target_amplitude(final_numpy[:, ch], original_audio_numpy)
        else:
            final_numpy = match_target_amplitude(final_numpy, original_audio_numpy)
        final_numpy = np.clip(final_numpy, -1.0, 1.0)
    
    output_path = SPACE_ROOT / "output.flac"
    sf.write(output_path, final_numpy, target_fs)
    
    return output_path.as_posix(), f"Done! Seed: {seed}"

# --- UI ---
css = ".container { max-width: 800px; margin: auto; }"
with gr.Blocks(title="SonicMaster Colab", theme=gr.themes.Soft(), css=css) as app:
    gr.Markdown("# 🎛️ SonicMaster: AI Music Restoration")
    
    with gr.Group():
        default_token = os.getenv("HF_TOKEN") or ""
        hf_token_inp = gr.Textbox(label="Hugging Face Token", value=default_token, type="password")
        load_btn = gr.Button("🔌 Load Model", variant="primary")
        load_status = gr.Textbox(label="Status", interactive=False, value="Waiting...")

    input_audio = gr.Audio(label="Source Audio", type="numpy")
    prompt = gr.Textbox(label="Prompt", value="Master this track, remove reverb, high fidelity")
    
    with gr.Accordion("⚙️ Advanced Settings", open=True):
        with gr.Row():
            solver_drop = gr.Dropdown(choices=["Euler", "rk4"], value="Euler", label="Solver (rk4 is slower but cleaner)")
            full_song_chk = gr.Checkbox(label="Full Song Mode", value=True)
            normalize_chk = gr.Checkbox(label="Match Input Volume", value=True)
        steps = gr.Slider(10, 150, value=50, step=1, label="Inference Steps (Default: 50)")
        cfg = gr.Slider(1.0, 10.0, value=3.0, step=0.1, label="Guidance Scale")
        seed = gr.Number(label="Seed (-1 = Random)", value=-1, precision=0)
    
    run_btn = gr.Button("🚀 Process Audio", variant="primary", interactive=False, size="lg")
    output_audio = gr.Audio(label="Restored Audio", type="filepath")
    result_info = gr.Textbox(label="Processing Info", interactive=False)

    def unlock_ui(status):
        return gr.update(interactive=True) if "successfully" in status else gr.update(interactive=False)

    load_btn.click(load_models, inputs=[hf_token_inp], outputs=[load_status]).then(
        unlock_ui, inputs=[load_status], outputs=[run_btn]
    )
    
    run_btn.click(
        process_audio,
        inputs=[input_audio, prompt, steps, cfg, seed, solver_drop, full_song_chk, normalize_chk],
        outputs=[output_audio, result_info]
    )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)