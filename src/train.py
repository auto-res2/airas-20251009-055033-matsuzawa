#!/usr/bin/env python
"""
Training / adaptation runner for a SINGLE experiment variation.
This file is launched by src.main for each run.
"""
from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import yaml
from torch import optim
from tqdm import tqdm

from .preprocess import get_dataloaders, set_seed
from .model import build_model


################################################################################
# ------------------------------  ADAPTERS  ---------------------------------- #
################################################################################

class _ActivationCacher:
    """Registers forward hooks on affine normalisation layers and stores outputs.
    The collected activations are needed for forward-only adaptation algorithms
    that rely on statistics of the current batch (e.g. NGFAT / ZORRO).
    """

    def __init__(self, model: nn.Module):
        self.affine: List[nn.Module] = []
        self.cache: Dict[int, torch.Tensor] = {}
        for m in model.modules():
            if isinstance(
                m,
                (
                    nn.BatchNorm2d,
                    nn.BatchNorm1d,
                    nn.GroupNorm,
                    nn.LayerNorm,
                ),
            ):
                self.affine.append(m)
                m.register_forward_hook(self._make_hook(m))

    def _make_hook(self, module: nn.Module):
        def _hook(_, __, output):
            self.cache[id(module)] = output.detach()

        return _hook


# -----------------------------------------------------------------------------
class NGFATState(_ActivationCacher):
    """Simplified NGFAT implementation: always applies a natural-gradient step on
    γ, β of normalisation layers. Only diagonal Fisher ≈ Var[y] is used.
    """

    def __init__(self, model: nn.Module, lr: float = 1e-3):
        super().__init__(model)
        self.lr = lr

    @torch.no_grad()
    def step(self, logits: torch.Tensor):
        for mod in self.affine:
            y = self.cache[id(mod)]
            if y.ndim > 2:
                g = y.mean(dim=(0, 2, 3))  # gradient proxy wrt γ
                fisher = y.var(dim=(0, 2, 3)) + 1e-5
            else:
                g = y.mean(dim=0)
                fisher = y.var(dim=0) + 1e-5
            update = -self.lr * g / fisher
            mod.weight.add_(update.view_as(mod.weight))


# -----------------------------------------------------------------------------
class RoTTAState(_ActivationCacher):
    """Very light-weight risk-aware gate: similar to NGFAT but only updates when
    batch entropy is above a threshold δ (default 0.1).
    """

    def __init__(self, model: nn.Module, lr: float = 1e-3, delta: float = 0.1):
        super().__init__(model)
        self.lr = lr
        self.delta = delta

    @torch.no_grad()
    def step(self, logits: torch.Tensor):
        probs = torch.softmax(logits, dim=1)
        entropy = (-probs * torch.log_softmax(logits, dim=1)).sum(1).mean()
        if entropy < self.delta:
            return  # skip update if batch looks easy
        # else identical to NGFAT
        for mod in self.affine:
            y = self.cache[id(mod)]
            if y.ndim > 2:
                g = y.mean(dim=(0, 2, 3))
                fisher = y.var(dim=(0, 2, 3)) + 1e-5
            else:
                g = y.mean(dim=0)
                fisher = y.var(dim=0) + 1e-5
            update = -self.lr * g / fisher
            mod.weight.add_(update.view_as(mod.weight))


# -----------------------------------------------------------------------------
class ZorroState(_ActivationCacher):
    """ZORRO implementation with optional rollback (disabled in lite version)."""

    def __init__(
        self,
        model: nn.Module,
        lambda_: float = 1.0,
        eps: float = 1e-3,
        k_ckpt: int = 3,
        enable_rollback: bool = True,
    ):
        super().__init__(model)
        self.lambda_ = lambda_
        self.eps = eps
        self.n = 0  # effective sample count
        self.enable_rollback = enable_rollback
        self.last_acc_hat = 1.0
        self.last_entropy = 0.0
        self.checkpoints: deque = deque(maxlen=k_ckpt if enable_rollback else 1)

    # ---------------------------------------------------------------------
    @torch.no_grad()
    def step(self, logits: torch.Tensor):
        probs = torch.softmax(logits, dim=1)
        log_probs = torch.log_softmax(logits, dim=1)
        entropy = -(probs * log_probs).sum(1)
        var_proxy = (probs * (1 - probs)).sum(1)
        acc_hat = 1.0 - var_proxy.mean()

        should_update = (
            (acc_hat < self.last_acc_hat - self.eps)
            or (entropy.mean() > self.last_entropy * 0.9)
        )

        tau = self.n / (self.n + self.lambda_) if (self.n + self.lambda_) > 0 else 0.0

        if should_update:
            for mod in self.affine:
                y = self.cache[id(mod)]
                if y.ndim > 2:
                    g = y.mean(dim=(0, 2, 3))
                    fisher = y.var(dim=(0, 2, 3)) + 1e-5
                else:
                    g = y.mean(dim=0)
                    fisher = y.var(dim=0) + 1e-5
                step = -tau * g / fisher
                mod.weight.add_(step.view_as(mod.weight))
            if self.enable_rollback:
                self.checkpoints.append((self._snapshot(), acc_hat.item()))
        else:
            if (
                self.enable_rollback
                and len(self.checkpoints) == self.checkpoints.maxlen
                and acc_hat.item() > max(a for _, a in self.checkpoints)
            ):
                best_state, _ = max(self.checkpoints, key=lambda x: x[1])
                self._restore(best_state)

        self.last_acc_hat = acc_hat.item()
        self.last_entropy = entropy.mean().item()
        self.n += 1

    # ------------------------------------------------------------------ utils
    def _snapshot(self):
        return {
            id(m): {
                "weight": m.weight.data.clone(),
                "bias": None if m.bias is None else m.bias.data.clone(),
            }
            for m in self.affine
        }

    def _restore(self, state_dict):
        for m in self.affine:
            buf = state_dict[id(m)]
            m.weight.data.copy_(buf["weight"])
            if m.bias is not None and buf["bias"] is not None:
                m.bias.data.copy_(buf["bias"])


################################################################################
# ------------------------------  TRAIN LOGIC  ------------------------------- #
################################################################################

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config-path", type=str, required=True)
    p.add_argument("--results-dir", type=str, required=True)
    return p.parse_args()


def _create_adapter(method: str, model: nn.Module, training_cfg: dict):
    method = method.lower()
    if method == "ngfat":
        return NGFATState(model, lr=float(training_cfg.get("adapt_lr", 1e-3)))
    if method == "rotta":
        return RoTTAState(
            model,
            lr=float(training_cfg.get("adapt_lr", 1e-3)),
            delta=float(training_cfg.get("delta", 0.1)),
        )
    if method in {"zorro-lite", "zorro_full", "zorro-full", "zorro"}:
        return ZorroState(
            model,
            lambda_=float(training_cfg.get("lambda", 1.0)),
            eps=float(training_cfg.get("eps", 1e-3)),
            k_ckpt=3,
            enable_rollback=method not in {"zorro-lite"},
        )
    return None  # source / frozen – no adapter


def main():
    args = parse_args()

    cfg = yaml.safe_load(Path(args.config_path).read_text())
    run_id = cfg["run_id"]

    results_root = Path(args.results_dir)
    results_root.mkdir(parents=True, exist_ok=True)

    # ------------------- reproducibility & device -------------------- #
    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------- data -------------------------------- #
    train_loader, val_loader, num_classes = get_dataloaders(
        cfg["dataset"], cfg["training"]
    )

    # --------------------------- model ------------------------------- #
    model_cfg = cfg["model"]
    model_cfg["num_classes"] = num_classes
    model = build_model(model_cfg).to(device)

    if model_cfg.get("pretrained"):
        model.load_state_dict(torch.load(model_cfg["pretrained"], map_location=device))

    # --------------------- method & optimiser ----------------------- #
    method = cfg["training"].get("method", "source").lower()
    optimiser = optim.Adam(model.parameters(), lr=cfg["training"].get("learning_rate", 1e-3))
    criterion = nn.CrossEntropyLoss()

    adapter = _create_adapter(method, model, cfg["training"])

    # Decide training / eval mode: only "source" uses back-prop.
    requires_grad = method == "source"
    model.train(requires_grad)

    epochs = cfg["training"].get("epochs", 1)
    history: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        correct = total = 0
        pbar = tqdm(train_loader, desc=f"[{run_id}] Epoch {epoch}/{epochs}")
        for batch in pbar:
            inputs, targets = (b.to(device) for b in batch)

            if adapter is not None:
                # ---------------- forward pass ---------------- #
                with torch.no_grad():
                    logits = model(inputs)
                    adapter.step(logits)
                loss = criterion(logits, targets)
            elif method == "source-frozen":
                with torch.no_grad():
                    logits = model(inputs)
                loss = criterion(logits, targets)
            else:  # Supervised source training
                optimiser.zero_grad()
                logits = model(inputs)
                loss = criterion(logits, targets)
                loss.backward()
                optimiser.step()

            # ---------------- metrics ------------------------ #
            epoch_loss += loss.item() * inputs.size(0)
            _, preds = logits.max(1)
            correct += preds.eq(targets).sum().item()
            total += inputs.size(0)
            pbar.set_postfix({"loss": loss.item(), "acc": 100.0 * correct / total})

        history.append({
            "epoch": epoch,
            "loss": epoch_loss / total,
            "acc": 100.0 * correct / total,
        })

    # --------------------------- validation ------------------------- #
    model.eval()
    val_correct = val_total = 0
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss = criterion(logits, targets)
            val_loss += loss.item() * inputs.size(0)
            _, preds = logits.max(1)
            val_correct += preds.eq(targets).sum().item()
            val_total += inputs.size(0)
    val_loss /= val_total
    val_acc = 100.0 * val_correct / val_total

    # ---------------------------- save ------------------------------ #
    torch.save(model.state_dict(), results_root / "model.pt")
    metrics = {
        "run_id": run_id,
        "method": method,
        "epochs": epochs,
        "train_history": history,
        "val_loss": val_loss,
        "val_acc": val_acc,
    }
    (results_root / "results.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()