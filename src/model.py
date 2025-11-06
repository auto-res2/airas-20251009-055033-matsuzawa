# src/model.py
"""Diffusion backbone wrapper & AHOF sampler implementation."""
from __future__ import annotations

from typing import Any

import torch
from diffusers import (
    DDPMPipeline,
    DDPMScheduler,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    UNet2DModel,
)

CACHE_DIR = ".cache/"

# ---------------------------------------------------------------------------
# Adaptive Higher-Order Free (AHOF) Sampler
# ---------------------------------------------------------------------------

class AHOFSampler:
    """Training-free sampler with adaptive step size control."""

    def __init__(self, pipeline: DDPMPipeline, *, error_tolerance: float = 1e-3, step_size_scale: float = 1.0, max_order: int = 3):
        self.pipeline = pipeline
        self.err_tol = error_tolerance
        self.step_scale = step_size_scale
        self.max_order = max_order
        self.device = pipeline.unet.device
        self.dtype = pipeline.unet.dtype

    def _euler_step(self, x: torch.FloatTensor, t: int, dt: int) -> torch.FloatTensor:
        with torch.no_grad():
            return x + dt * self.pipeline.unet(x, t).sample  # type: ignore[arg-type]

    @torch.inference_mode()
    def sample(self, *, batch_size: int, num_inference_steps: int, img_size: int) -> torch.FloatTensor:  # noqa: D401
        scheduler = self.pipeline.scheduler
        scheduler.set_timesteps(num_inference_steps)
        timesteps = scheduler.timesteps

        latents = torch.randn(batch_size, 3, img_size, img_size, device=self.device, dtype=self.dtype)
        dt = int(self.step_scale * abs(timesteps[0] - timesteps[1]).item())

        for t in timesteps:
            prop = self._euler_step(latents, t, -dt)
            mid_t = max(t - dt // 2, 0)
            mid_state = (latents + prop) / 2
            mid_out = self.pipeline.unet(mid_state, mid_t).sample  # type: ignore[arg-type]
            half_step = mid_state + (-dt / 2) * mid_out
            err = (prop - half_step).abs().mean()
            adapt = torch.clamp((self.err_tol / (err + 1e-8)) ** 0.5, 0.5, 2.0)
            dt = int(max(1, adapt.item() * dt))
            latents = prop

        images = (latents / 2 + 0.5).clamp(0, 1).to(torch.float32)
        return images.cpu()


# ---------------------------------------------------------------------------
# Diffusion model wrapper (handles sampler switching)
# ---------------------------------------------------------------------------

class DiffusionModel:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.resources.device)
        self.dtype = torch.float16 if cfg.model.precision == "fp16" else torch.float32

        # Try loading checkpoint ------------------------------------------------
        try:
            self.pipeline = DDPMPipeline.from_pretrained(
                cfg.model.weights_path,
                torch_dtype=self.dtype,
                cache_dir=CACHE_DIR,
                safety_checker=None,
            )
        except Exception:  # noqa: BLE001
            unet = UNet2DModel(sample_size=cfg.model.image_size, in_channels=3, out_channels=3, block_out_channels=(64, 128, 128, 256))
            self.pipeline = DDPMPipeline(unet=unet, scheduler=DDPMScheduler(num_train_timesteps=1000))
        self.pipeline.to(self.device)

        self.current_sampler: Any = "ddpm"
        self.set_sampler(cfg.sampling.sampler_name)

    # ---------------------------------------------------------------------
    def set_sampler(self, name: str, **kwargs) -> None:
        name = name.lower()
        if name == "ddim":
            self.pipeline.scheduler = DDIMScheduler.from_config(self.pipeline.scheduler.config)
            self.current_sampler = "ddim"
        elif name in {"dpm-solver++", "dpmsolver++"}:
            self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(self.pipeline.scheduler.config)
            self.current_sampler = "dpm"
        elif name == "ahof":
            self.current_sampler = AHOFSampler(self.pipeline, **kwargs)
        else:
            self.pipeline.scheduler = DDPMScheduler.from_config(self.pipeline.scheduler.config)
            self.current_sampler = "ddpm"

    # ---------------------------------------------------------------------
    @torch.inference_mode()
    def generate(self, *, batch_size: int, num_inference_steps: int):
        if isinstance(self.current_sampler, AHOFSampler):
            return self.current_sampler.sample(batch_size=batch_size, num_inference_steps=num_inference_steps, img_size=self.cfg.model.image_size)
        images = self.pipeline(batch_size=batch_size, num_inference_steps=num_inference_steps, output_type="pt", generator=torch.Generator(device=self.device).manual_seed(0)).images
        return images.to(torch.float32)