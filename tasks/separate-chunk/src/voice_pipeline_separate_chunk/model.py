from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


class DialogueSidon:
    def __init__(self, policy, token=None):
        self.policy = policy
        self.token = token
        self._models = None

    def _load(self):
        if self._models is not None:
            return self._models
        import torch
        from diffusers import DPMSolverMultistepScheduler
        from huggingface_hub import hf_hub_download

        selected = (
            "cuda"
            if self.policy.device == "auto" and torch.cuda.is_available()
            else ("cpu" if self.policy.device == "auto" else self.policy.device)
        )
        device = torch.device(selected)
        names = (
            "ssl_encoder.pt2",
            "diffusion_head.pt2",
            "vae_decoder.pt2",
            "metadata.json",
        )
        paths = {
            n: hf_hub_download(
                self.policy.repo_id, n, revision=self.policy.revision, token=self.token
            )
            for n in names
        }
        expected = {
            "ssl_encoder.pt2": (
                "295125ed72772ca4cf87c3dacbdd74019a3d4493356945e139ca642ca2e1e639",
                1352938501,
            ),
            "diffusion_head.pt2": (
                "c44a09d5b00d08d6534121da1075bc36b80673d1361ca14355e73f1b9c0809fd",
                354885536,
            ),
            "vae_decoder.pt2": (
                "693c0622f4d032bb2fbac044d6a42e5fbebfb05649f78fe04c6bcb6d17b8c316",
                167903493,
            ),
            "metadata.json": (
                "c5d017bf92c6d7c8656e398aab11bc4a99381ede875a84e22ae01451eba38757",
                4163,
            ),
        }
        for name, path in paths.items():
            artifact = Path(path)
            hasher = hashlib.sha256()
            with artifact.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    hasher.update(block)
            digest = hasher.hexdigest()
            if (
                artifact.stat().st_size != expected[name][1]
                or digest != expected[name][0]
            ):
                raise RuntimeError("DialogueSidon artifact verification failed")
        with open(paths["metadata.json"], encoding="utf-8") as stream:
            meta = json.load(stream)
        models = {
            "ssl": torch.export.load(paths["ssl_encoder.pt2"]).module().to(device),
            "head": torch.export.load(paths["diffusion_head.pt2"]).module().to(device),
            "vae": torch.export.load(paths["vae_decoder.pt2"]).module().to(device),
            "meta": meta,
            "device": device,
        }
        models["scheduler"] = DPMSolverMultistepScheduler.from_config(
            meta["ddpm_config"],
            algorithm_type="dpmsolver++",
            timestep_spacing="linspace",
        )
        models["mean"] = torch.tensor(meta["latent_norm_mean"], device=device).view(
            1, 1, -1
        )
        models["std"] = torch.tensor(meta["latent_norm_std"], device=device).view(
            1, 1, -1
        )
        self._models = models
        return models

    def separate(self, samples: np.ndarray, *, seed: int) -> tuple[np.ndarray, int]:
        import torch
        import torchaudio

        m = self._load()
        device = m["device"]
        wav = torch.as_tensor(samples, dtype=torch.float32, device=device)
        maximum = wav.abs().max().clamp_min(1e-6)
        wav = torch.nn.functional.pad((0.9 * wav / maximum), (160, 160))
        feat = torchaudio.compliance.kaldi.fbank(
            wav[None, :],
            sample_frequency=16000,
            num_mel_bins=80,
            frame_length=25,
            frame_shift=10,
            dither=0.0,
            preemphasis_coefficient=0.97,
            remove_dc_offset=True,
            window_type="povey",
            use_energy=False,
            energy_floor=1.192092955078125e-07,
        )
        feat = (feat - feat.mean(0, keepdim=True)) / torch.sqrt(
            feat.var(0, keepdim=True) + 1e-5
        )
        length = (feat.shape[0] // 2) * 2
        feat = feat[:length].reshape(1, length // 2, 160)
        mask = torch.ones((1, length // 2), dtype=torch.int64, device=device)
        with torch.inference_mode():
            features, p0, p1 = m["ssl"](feat, mask)
            predicted = torch.cat((p0, p1), -1)
            if m["meta"]["latent_norm_initialized"]:
                predicted = (predicted - m["mean"]) / m["std"]
            conditioning = torch.cat((predicted, features), -1)
            dim = m["meta"]["latent_dim"]
            generator = torch.Generator(device=device).manual_seed(seed)
            latents = torch.randn(
                (1, conditioning.shape[1], dim * 2),
                generator=generator,
                device=device,
                dtype=conditioning.dtype,
            )
            scheduler = m["scheduler"]
            scheduler.set_timesteps(self.policy.inference_steps, device=device)
            for t in scheduler.timesteps:
                tb = torch.full((1,), int(t.item()), device=device, dtype=torch.long)
                latents = scheduler.step(
                    m["head"](latents, tb, conditioning), t, latents
                ).prev_sample
            if m["meta"]["latent_norm_initialized"]:
                latents = latents * m["std"] + m["mean"]
            tracks = torch.cat(
                (
                    m["vae"](latents[:, :, :dim].transpose(1, 2)),
                    m["vae"](latents[:, :, dim:].transpose(1, 2)),
                ),
                0,
            ).squeeze(1)
        return tracks.float().cpu().numpy(), int(m["meta"]["sample_rate"])

    def close(self):
        self._models = None
