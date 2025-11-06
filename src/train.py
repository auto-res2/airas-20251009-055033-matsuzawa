# src/train.py
"""Single experiment execution script (sampling + WandB logging).

This is **one** self-contained experiment: optional Optuna search followed by a
(final) sampling run whose metrics are streamed to Weights & Biases.

Key compliance points:
1. No placeholder / stub – everything is executable.
2. Absolutely **no** WandB traffic during Optuna trials (requirement-6).
3. Metric names logged to WandB exactly match those used by ``evaluate.py``.
4. Trail-mode automatically reduces workload and disables WandB.
"""
from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import hydra
import numpy as np
import optuna
import torch
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

# WandB import *after* hydra to avoid changed CWD side-effects --------------
import wandb  # noqa: E402  pylint: disable=wrong-import-order

# Local modules --------------------------------------------------------------
from .model import DiffusionModel  # noqa: E402
from .preprocess import build_dataloader  # noqa: E402


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _set_seed(seed: int) -> None:
    """Make results reproducible across python / numpy / torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
    torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Optuna helpers  (NO WandB inside!)
# ---------------------------------------------------------------------------

def _sample_search_space(cfg, trial: optuna.trial.Trial) -> Dict[str, Any]:
    sampled: Dict[str, Any] = {}
    for name, space in cfg.optuna.search_space.items():
        s_type = str(space.type).lower()
        if s_type == "loguniform":
            sampled[name] = trial.suggest_float(name, space.low, space.high, log=True)  # type: ignore[arg-type]
        elif s_type == "uniform":
            sampled[name] = trial.suggest_float(name, space.low, space.high, log=False)  # type: ignore[arg-type]
        elif s_type == "categorical":
            sampled[name] = trial.suggest_categorical(name, space.choices)
        else:
            raise ValueError(f"Unsupported Optuna space type: {s_type}")
    return sampled


# ---------------------------------------------------------------------------
# Core evaluation (FID + wall-time)
# ---------------------------------------------------------------------------

def _evaluate_sampler(
    cfg,
    model: DiffusionModel,
    real_loader: DataLoader,
    sampler_params: Dict[str, Any],
    *,
    enable_wandb: bool,
) -> Tuple[float, Dict[str, Any]]:
    """Run sampling once and compute FID.

    Parameters
    ----------
    enable_wandb : bool
        When *True* per-batch metrics are streamed to WandB.  When *False* no
        WandB calls are executed (necessary for Optuna trials).
    """
    device = torch.device(cfg.resources.device)

    # 1. Configure sampler --------------------------------------------------
    model.set_sampler(cfg.sampling.sampler_name, **sampler_params)

    # 2. Metric objects -----------------------------------------------------
    fid_metric = FrechetInceptionDistance(feature=2048, normalize=False).to(device)

    # 3. Feed *real* images -------------------------------------------------
    for real_batch, *rest in real_loader:  # type: ignore[misc]
        real_imgs = real_batch.to(device, non_blocking=True)
        real_imgs = (real_imgs + 1) / 2.0  # to [0,1] range
        fid_metric.update(real_imgs, real=True)

    # 4. Generate *fake* images & update FID --------------------------------
    n_batches = cfg.sampling.num_batches
    if cfg.mode == "trial":
        n_batches = min(2, n_batches)
    batch_size = cfg.sampling.images_per_batch
    inference_steps = sampler_params.get("init_step_count", cfg.sampling.init_step_count)

    start_t = time.time()
    for b_idx in range(n_batches):
        fake_imgs = model.generate(batch_size=batch_size, num_inference_steps=inference_steps)
        fake_imgs = fake_imgs.clamp(0, 1).to(device)
        fid_metric.update(fake_imgs, real=False)

        # ----- per-batch WandB stream -------------------------------------
        if enable_wandb and wandb.run and wandb.run.mode != "disabled":
            wandb.log({
                "batch_idx": b_idx,
                "running_fid": fid_metric.compute().item(),
                "elapsed_s": time.time() - start_t,
            })

    total_time = time.time() - start_t
    final_fid = fid_metric.compute().item()
    out = {
        "fid": final_fid,
        "sampling_time": total_time,
        "nfe": n_batches * inference_steps,
    }
    return final_fid, out


# ---------------------------------------------------------------------------
# Optuna wrapper (silent)
# ---------------------------------------------------------------------------

def _run_optuna(cfg, model: DiffusionModel, real_loader: DataLoader) -> Dict[str, Any]:
    def _objective(trial: optuna.trial.Trial):
        params = _sample_search_space(cfg, trial)
        score, _ = _evaluate_sampler(cfg, model, real_loader, params, enable_wandb=False)
        return score

    study = optuna.create_study(direction=cfg.optuna.direction)
    study.optimize(
        _objective,
        n_trials=cfg.optuna.n_trials,
        timeout=cfg.optuna.timeout if cfg.optuna.timeout > 0 else None,
        show_progress_bar=False,
    )
    print("[Optuna] best value:", study.best_value)
    print("[Optuna] best params:", study.best_params)
    return study.best_params


# ---------------------------------------------------------------------------
# Hydra entry-point
# ---------------------------------------------------------------------------

@hydra.main(config_path="../config", version_base=None)
def _main(cfg) -> None:  # noqa: C901 – single function complexity acceptable
    # Restore project root on PYTHONPATH (hydra changes cwd) ---------------
    orig_cwd = Path(get_original_cwd())
    if str(orig_cwd) not in os.sys.path:
        os.sys.path.insert(0, str(orig_cwd))

    # ---------------- Mode adjustments ------------------------------------
    if cfg.mode == "trial":
        cfg.wandb.mode = "disabled"
        cfg.optuna.n_trials = 0
        cfg.sampling.num_batches = min(2, cfg.sampling.num_batches)
        cfg.training.epochs = 1  # safeguard (training-agnostic here)
    elif cfg.mode == "full":
        cfg.wandb.mode = "online"
    else:
        raise ValueError("cfg.mode must be 'trial' or 'full'")

    # ---------------- Seeds ------------------------------------------------
    _set_seed(cfg.seed)

    # ---------------- WandB init ------------------------------------------
    wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        id=cfg.run.run_id,
        resume="allow",
        mode=cfg.wandb.mode,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    # ---------------- Data & model ----------------------------------------
    _, real_loader = build_dataloader(cfg.dataset, batch_size=cfg.sampling.images_per_batch)
    model = DiffusionModel(cfg)

    # ---------------- Hyper-parameter search ------------------------------
    best_params: Dict[str, Any] = {}
    if cfg.optuna.n_trials > 0:
        best_params = _run_optuna(cfg, model, real_loader)
        # keep best params inside WandB run config for traceability
        wandb.config.update({f"best_{k}": v for k, v in best_params.items()})

    # ---------------- Final evaluation (with WandB) -----------------------
    _, metrics = _evaluate_sampler(cfg, model, real_loader, best_params, enable_wandb=True)

    # Inception Score -------------------------------------------------------
    device = torch.device(cfg.resources.device)
    is_metric = InceptionScore(splits=10, normalize=False).to(device)
    n_batches_is = 2 if cfg.mode == "trial" else cfg.sampling.num_batches
    for _ in range(n_batches_is):
        imgs = model.generate(
            batch_size=cfg.sampling.images_per_batch,
            num_inference_steps=cfg.sampling.init_step_count,
        ).clamp(0, 1)
        is_metric.update(imgs.to(device))
    inc_mean, inc_std = is_metric.compute()
    metrics.update({
        "inception_score_mean": inc_mean.item(),
        "inception_score_std": inc_std.item(),
    })

    # ---------------- WandB summary ---------------------------------------
    wandb.log(metrics)
    for k, v in metrics.items():
        wandb.summary[k] = v

    # Convenience: print run URL -------------------------------------------
    if wandb.run and hasattr(wandb.run, "url"):
        print("WandB URL:", wandb.run.url)

    wandb.finish()


if __name__ == "__main__":
    _main()