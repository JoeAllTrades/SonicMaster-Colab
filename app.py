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

    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    # Weights resolution
    possible_paths = [
        SPACE_ROOT / "model.safetensors",
        SPACE_ROOT / "SonicMaster" / "model.safetensors",
        SPACE_ROOT / "weights" / "model.safetensors"
    ]
    model_path = next((p for p in possible_paths if p.exists()), None)
            
    if not model_path:
         from huggingface_hub import hf_hub_download
         model_path = hf_hub_download(repo_id="amaai-lab/SonicMaster", filename="model.safetensors")

    # Load TangoFlux (MM-DiT + DiT)
    model = TangoFlux(config=config["model"])
    weights = load_file(str(model_path))
    model.load_state_dict(weights, strict=False)
    model.to(DEVICE).eval()
    
    for param in model.text_encoder.parameters():
        param.requires_grad = False
    MODEL = model

    # Load Stable Audio Open VAE (required by paper)
    try:
        vae = AutoencoderOobleck.from_pretrained(
            "stabilityai/stable-audio-open-1.0", 
            subfolder="vae",
            token=hf_token
        ).to(DEVICE)
        vae.eval()
        VAE = vae
    except Exception as e:
        return f"❌ VAE Load Error: {e}"

    return "✅ SonicMaster loaded successfully!"

def match_target_amplitude(sound, target_sound, epsilon=1e-4):
    def rms(x): return np.sqrt(np.mean(x**2))
    source_rms = rms(sound)
    target_rms = rms(target_sound)
    if source_rms < epsilon: return sound
    gain = target_rms / (source_rms + epsilon)
    return sound * min(gain, 5.0)

def process_audio(input_audio, prompt, steps, cfg, seed, solver, ensemble, normalize):
    if MODEL is None or VAE is None:
        raise gr.Error("Please load the model first!")
    
    # Seeding
    if seed == -1: seed = random.randint(0, 2**32 - 1)
    torch.manual_seed(seed)
    random.seed(seed)
    
    sr, audio_data = input_audio
    target_fs = 44100
    
    # Prepare Audio Tensor [C, T]
    if audio_data.dtype == np.int16: audio_data = audio_data / 32768.0
    elif audio_data.dtype == np.int32: audio_data = audio_data / 2147483648.0
    audio_tensor = torch.from_numpy(audio_data).float()
    orig_np = audio_tensor.numpy().flatten()

    if audio_tensor.ndim == 1: audio_tensor = audio_tensor.unsqueeze(0).repeat(2, 1)
    elif audio_tensor.ndim == 2: audio_tensor = audio_tensor.t()
    
    if sr != target_fs:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_fs)
        audio_tensor = resampler(audio_tensor)
    
    audio_tensor = audio_tensor.to(DEVICE)

    # Paper Specs: 30s chunks, 10s overlap for conditioning
    chunk_dur = 30
    overlap_dur = 10
    chunk_size = chunk_dur * target_fs
    overlap_size = overlap_dur * target_fs
    stride = chunk_size - overlap_size

    chunks = []
    start = 0
    while start < audio_tensor.shape[1]:
        end = min(start + chunk_size, audio_tensor.shape[1])
        chunk = audio_tensor[:, start:end]
        if chunk.shape[1] < chunk_size:
            chunk = torch.nn.functional.pad(chunk, (0, chunk_size - chunk.shape[1]))
        chunks.append(chunk)
        start += stride

    # Latent encoding (Batched VAE)
    degraded_latents = []
    with torch.no_grad():
        stack = torch.stack(chunks).to(DEVICE)
        for b in range(0, len(stack), 4):
            lat = VAE.encode(stack[b:b+4]).latent_dist.mode()
            degraded_latents.append(lat)
        degraded_latents = torch.cat(degraded_latents, dim=0)

    decoded_waves = []
    prev_out_cue = None # The 10s 'audio pooling' reference from previous output

    for i in range(len(degraded_latents)):
        # Paper: "The last 10s of this output are used to condition the next segment"
        # We re-encode the previous output segment to get the conditioning latent
        cond_latent = None
        if i > 0 and prev_out_cue is not None:
            with torch.no_grad():
                # Encode last 10s of previous output as the clean reference
                cond_latent = VAE.encode(prev_out_cue.to(DEVICE)).latent_dist.mode().transpose(1, 2)

        ensemble_accum = None
        for e in range(ensemble):
            current_seed = seed + i + (e * 555)
            torch.manual_seed(current_seed)
            
            z_in = degraded_latents[i].unsqueeze(0).transpose(1, 2)
            
            with torch.no_grad():
                res_latent = MODEL.inference_flow(
                    z_in, prompt,
                    audiocond_latents=cond_latent,
                    num_inference_steps=steps,
                    guidance_scale=cfg,
                    duration=chunk_dur,
                    seed=current_seed,
                    solver=solver
                )
                decoded = VAE.decode(res_latent.transpose(2, 1)).sample.cpu()
                if ensemble_accum is None: ensemble_accum = decoded
                else: ensemble_accum += decoded
        
        chunk_out = ensemble_accum / ensemble
        decoded_waves.append(chunk_out)
        
        # Save last 10s for next chunk conditioning (Audio Pooling Branch)
        prev_out_cue = chunk_out[:, :, -overlap_size:]
        torch.cuda.empty_cache()

    # Stitching (Linear interpolation over 10s as per paper)
    final_audio = decoded_waves[0]
    alpha = torch.linspace(1, 0, steps=overlap_size).view(1, 1, -1)
    beta = 1 - alpha

    for i in range(1, len(decoded_waves)):
        prev_part = final_audio[:, :, -overlap_size:]
        curr_part = decoded_waves[i][:, :, :overlap_size]
        blended = prev_part * alpha + curr_part * beta
        final_audio = torch.cat([final_audio[:, :, :-overlap_size], blended, decoded_waves[i][:, :, overlap_size:]], dim=2)
    
    final_np = final_audio.squeeze(0).numpy().T
    if normalize:
        for ch in range(final_np.shape[1]):
            final_np[:, ch] = match_target_amplitude(final_np[:, ch], orig_np)
        final_np = np.clip(final_np, -1.0, 1.0)
    
    out_path = SPACE_ROOT / "sonic_output.wav"
    sf.write(out_path, final_np, target_fs, subtype='FLOAT')
    return out_path.as_posix(), f"Success. Seed: {seed}"

# --- UI Construction ---
with gr.Blocks(title="SonicMaster PRO", theme=gr.themes.Default()) as app:
    gr.Markdown("# 🎧 SonicMaster: Controllable Music Restoration")
    gr.Markdown("Based on *'SonicMaster: Towards Controllable All-in-One Music Restoration and Mastering'* (2025).")
    
    with gr.Row():
        with gr.Column():
            token = gr.Textbox(label="HF Token", type="password", value=os.getenv("HF_TOKEN", ""))
            load_btn = gr.Button("🚀 Step 1: Load Model")
            status = gr.Textbox(label="Status", interactive=False)
            
            in_audio = gr.Audio(label="Step 2: Upload Music", type="numpy")
            prompt = gr.Textbox(
                label="Step 3: Instructions (from Paper)", 
                placeholder="e.g. 'Remove the excess reverb', 'Fix the clipping and harshness'",
                value="Master this song, remove room echo, improve clarity"
            )
            
            with gr.Accordion("Advanced Paper Settings", open=False):
                solver = gr.Dropdown(["Euler", "rk4"], value="rk4", label="ODE Solver (rk4 = Paper High Quality)")
                steps = gr.Slider(1, 100, value=10, step=1, label="Steps (Paper uses 10 for rk4)")
                cfg = gr.Slider(1.0, 5.0, value=2.0, step=0.1, label="CFG (Keep low for restoration)")
                ensemble = gr.Slider(1, 4, value=1, step=1, label="Ensemble (Quality Multiplier)")
                norm = gr.Checkbox(label="Match Input Loudness", value=True)
                seed = gr.Number(label="Seed (-1 for Random)", value=-1)

            run_btn = gr.Button("✨ Process Entire Track", variant="primary", interactive=False)

        with gr.Column():
            out_audio = gr.Audio(label="Restored Result (32-bit WAV)")
            info = gr.Textbox(label="Log")

    def unlock(s): return gr.update(interactive="successfully" in s)
    load_btn.click(load_models, [token], [status]).then(unlock, [status], [run_btn])
    run_btn.click(process_audio, [in_audio, prompt, steps, cfg, seed, solver, ensemble, norm], [out_audio, info])

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)