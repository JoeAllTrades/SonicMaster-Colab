# inference_fullsong.py
import argparse
import os
from pathlib import Path
from time import time

import torch
import torchaudio
import soundfile as sf
import yaml
from safetensors.torch import load_file
from diffusers import AutoencoderOobleck

# Local imports (repo root is on sys.path when this file is executed)
from model import TangoFlux


def parse_args():
    p = argparse.ArgumentParser("Single-sample inference for SonicMaster")
    p.add_argument("--ckpt", type=str, required=True,
                   help="Path to model.safetensors (or directory containing it).")
    p.add_argument("--input", type=str, required=True,
                   help="Path to degraded input audio (wav/flac/etc).")
    p.add_argument("--prompt", type=str, required=True,
                   help="Text prompt guiding the enhancement/restoration.")
    p.add_argument("--output", type=str, required=True,
                   help="Output audio path (use .wav or .flac extension).")

    # Optional knobs (safe defaults)
    p.add_argument("--config", type=str, default=str(Path(__file__).parent / "configs" / "tangoflux_config.yaml"),
                   help="YAML config defining model sizes/hparams.")
    p.add_argument("--fs", type=int, default=44100, help="Target sample rate.")
    p.add_argument("--chunk_duration", type=int, default=30, help="Chunk length in seconds.")
    p.add_argument("--overlap_duration", type=int, default=10, help="Overlap (and carry) in seconds.")
    p.add_argument("--vae_batch_size", type=int, default=10, help="Batch size for VAE encoding over chunks.")
    p.add_argument("--num_inference_steps", type=int, default=10)
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument("--solver", type=str, default="Euler")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


@torch.no_grad()
def main():
    t0 = time()
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # --------- Resolve checkpoint path (file or directory) ----------
    ckpt_path = Path(args.ckpt)
    if ckpt_path.is_dir():
        candidate = ckpt_path / "model.safetensors"
        if not candidate.exists():
            raise FileNotFoundError(f"Could not find model.safetensors in {ckpt_path}")
        ckpt_path = candidate
    elif not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # --------- Load config & model ----------
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    model = TangoFlux(config=cfg["model"])

    weights = load_file(str(ckpt_path))
    model.load_state_dict(weights, strict=False)
    model.to(device).eval()

    # Freeze text encoder params
    for p in model.text_encoder.parameters():
        p.requires_grad = False
    model.text_encoder.eval()

    # --------- Load VAE ----------
    vae = AutoencoderOobleck.from_pretrained(
        "stabilityai/stable-audio-open-1.0", subfolder="vae"
    ).to(device)
    vae.eval()

    # --------- Read & standardize input ----------
    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input audio not found: {in_path}")

    audio, sr = torchaudio.load(str(in_path))  # [C, T]
    # Force stereo
    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)
    elif audio.shape[0] > 2:
        audio = audio[:2, :]

    # Resample to target fs
    if sr != args.fs:
        audio = torchaudio.functional.resample(audio, sr, args.fs)
        sr = args.fs

    audio = audio.to(device)

    # --------- Chunking ----------
    fs = args.fs
    chunk_size = args.chunk_duration * fs
    overlap = args.overlap_duration * fs
    if overlap <= 0 or overlap >= chunk_size:
        raise ValueError("overlap_duration must be >0 and smaller than chunk_duration.")
    stride = chunk_size - overlap

    chunks = []
    start = 0
    T = audio.shape[1]
    while start < T:
        end = min(start + chunk_size, T)
        ch = audio[:, start:end]
        if ch.shape[1] < chunk_size:
            ch = torch.nn.functional.pad(ch, (0, chunk_size - ch.shape[1]))
        chunks.append(ch)
        start += stride

    if not chunks:
        raise RuntimeError("No audio content to process.")

    # --------- Pre-encode degraded chunks with VAE (batched) ----------
    chunk_tensor = torch.stack(chunks)  # [N, 2, T]
    latents = []
    for b in range(0, chunk_tensor.shape[0], args.vae_batch_size):
        batch = chunk_tensor[b:b + args.vae_batch_size].to(device)
        z = vae.encode(batch).latent_dist.mode()  # [B, C, T']
        latents.append(z)
    degraded_latents = torch.cat(latents, dim=0)  # [N, C, T']

    # --------- Inference loop with conditional carry ----------
    decoded_chunks = []
    prev_cond = None
    g = torch.Generator(device=device).manual_seed(args.seed)

    for i in range(degraded_latents.shape[0]):
        # model expects [1, T', C] (transpose from [C, T'] if needed by your impl)
        z_in = degraded_latents[i].unsqueeze(0).transpose(1, 2)  # [1, T', C]

        result_latent = model.inference_flow(
            z_in,
            args.prompt,
            audiocond_latents=prev_cond,        # None for first chunk
            num_inference_steps=args.num_inference_steps,
            timesteps=None,
            guidance_scale=args.guidance_scale,
            duration=args.chunk_duration,
            seed=args.seed,
            disable_progress=True,
            num_samples_per_prompt=1,
            callback_on_step_end=None,
            solver=args.solver,
        )

        # Decode to waveform on CPU for stitching
        wav = vae.decode(result_latent.transpose(2, 1)).sample.cpu()  # [1, 2, T]
        # Safety clamp to [-1, 1]
        wav = torch.clamp(wav, -1.0, 1.0)
        decoded_chunks.append(wav)

        # Carry last overlap as conditioning (back on device)
        last = wav[:, :, -overlap:].to(device)
        prev_cond = vae.encode(last).latent_dist.mode().transpose(1, 2)  # [1, T', C]

    # --------- Crossfade stitch ----------
    final = decoded_chunks[0]  # [1, 2, T]
    for i in range(1, len(decoded_chunks)):
        prev = final[:, :, -overlap:]
        curr = decoded_chunks[i][:, :, :overlap]
        alpha = torch.linspace(1.0, 0.0, steps=overlap).view(1, 1, -1)
        beta = 1.0 - alpha
        blended = prev * alpha + curr * beta
        final = torch.cat(
            [final[:, :, :-overlap], blended, decoded_chunks[i][:, :, overlap:]],
            dim=2,
        )

    # --------- Save (honor extension) ----------
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = final.squeeze(0).numpy().T  # [T, 2]

    ext = out_path.suffix.lower()
    if ext == ".wav":
        sf.write(out_path.as_posix(), data, fs, format="WAV")
    elif ext == ".flac":
        sf.write(out_path.as_posix(), data, fs, format="FLAC")
    else:
        # Default to WAV if unknown extension
        sf.write(out_path.as_posix(), data, fs, format="WAV")

    print(f"Saved: {out_path}")
    print(f"Elapsed: {time() - t0:.2f}s")


if __name__ == "__main__":
    main()
