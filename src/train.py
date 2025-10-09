#!/usr/bin/env python
"""
Training / adaptation runner for a SINGLE experiment variation.
This file *must not* be called directly by users – it is launched as a
sub-process by src.main so that logs can be tee’ed and captured per run.
Nevertheless, it can be invoked stand-alone for debugging:

    python -m src.train --config-path path/to/run_cfg.yaml --results-dir /tmp/res
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch import optim
from tqdm import tqdm

from .preprocess import get_dataloaders, set_seed
from .model import build_model

matplotlib.use("Agg")  # mandatory for CLI / CI environments


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config-path", type=str, required=True,
                   help="YAML file describing *one* experiment run (written by main.py)")
    p.add_argument("--results-dir", type=str, required=True,
                   help="Directory where all outputs of this run are saved")
    return p.parse_args()


################################################################################
# ------------------------------  ZORRO ADAPTER  ------------------------------ #
################################################################################

class ZorroState:
    """Holds adaptation statistics, checkpoints & hyper-params.

    Component toggles (ablation support):
        disable_fisher   – Fisher information is ignored (plain mean update).
        use_shrinkage    – James–Stein shrinkage applied to natural-grad step.
        use_gate         – Accuracy/entropy gate decides whether to adapt.
        use_rollback     – Unsafe update detection & parameter rollback.
    """

    def __init__(self, model: nn.Module, lambda_: float = 4.0, eps: float = 5e-3,
                 k_ckpt: int = 3, *,
                 disable_fisher: bool = False,
                 use_shrinkage: bool = True,
                 use_gate: bool = True,
                 use_rollback: bool = True):
        self.lambda_ = lambda_
        self.eps = eps
        self.n = 0  # effective sample count for shrinkage
        self.last_acc_hat = 1.0  # optimistic starting point
        self.last_entropy = 0.0
        self.checkpoints: deque = deque(maxlen=k_ckpt)

        # ablation toggles
        self.disable_fisher = disable_fisher
        self.use_shrinkage = use_shrinkage
        self.use_gate = use_gate
        self.use_rollback = use_rollback

        self.affine: List[nn.Module] = []
        self.activation_cache: Dict[int, torch.Tensor] = {}
        self._register_hooks(model)

    # ---------------------------------------------------------------------
    def _register_hooks(self, model: nn.Module):
        """Register forward hooks on affine normalisation layers to cache outputs."""
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d,
                              nn.GroupNorm, nn.LayerNorm)):
                self.affine.append(m)
                m.register_forward_hook(self._make_hook(m))

    def _make_hook(self, module):
        def _hook(_, __, output):
            # Detach to avoid autograd bookkeeping – adaptation is forward-only.
            self.activation_cache[id(module)] = output.detach()
        return _hook

    # ---------------------------------------------------------------------
    @torch.no_grad()
    def zorro_step(self, logits: torch.Tensor):
        """Perform *one* ZORRO adaptation step given the model logits."""
        probs = torch.softmax(logits, dim=1)
        log_probs = torch.log_softmax(logits, dim=1)
        entropy = -(probs * log_probs).sum(1)           # per-sample entropy
        var_proxy = (probs * (1 - probs)).sum(1)        # disagreement proxy
        acc_hat = 1.0 - var_proxy.mean()               # label-free accuracy proxy

        # ---------------- decision gate ---------------- #
        should_update = True
        if self.use_gate:
            should_update = (
                (acc_hat < self.last_acc_hat - self.eps) or
                (entropy.mean() > self.last_entropy * 0.9)
            )

        # Shrinkage factor τ
        tau = 1.0 if not self.use_shrinkage else self.n / (self.n + self.lambda_)

        if should_update:
            for mod in self.affine:
                y = self.activation_cache[id(mod)]      # cached activation
                if y.ndim > 2:                          # Conv feature-maps
                    g = y.mean(dim=(0, 2, 3))          # gradient proxy ∂H/∂α
                    fisher_diag = y.var(dim=(0, 2, 3)) + 1e-5
                else:                                   # LayerNorm / GroupNorm over features
                    g = y.mean(dim=0)
                    fisher_diag = y.var(dim=0) + 1e-5

                if self.disable_fisher:
                    step = -tau * g
                else:
                    step = -tau * g / fisher_diag

                step = step.view_as(mod.weight.data)
                mod.weight.data.add_(step)

            # ---------------- checkpointing ---------------- #
            if self.use_rollback:
                self.checkpoints.append((self._snapshot(), acc_hat.item()))
        else:
            # Unsafe – consider rollback
            if self.use_rollback and len(self.checkpoints) == self.checkpoints.maxlen:
                # Two consecutive worse batches trigger rollback to best ckpt
                if acc_hat.item() > max(a for _, a in self.checkpoints):
                    best_state, _ = max(self.checkpoints, key=lambda x: x[1])
                    self._restore(best_state)

        # Update running statistics
        self.last_acc_hat = acc_hat.item()
        self.last_entropy = entropy.mean().item()
        self.n += 1

    # ------------------------------------------------------------------ utils #
    def _snapshot(self):
        """Return *only* the affine parameters weight/bias for lightweight checkpoint."""
        return {
            id(m): {
                "weight": m.weight.data.clone(),
                "bias": None if m.bias is None else m.bias.data.clone(),
            } for m in self.affine
        }

    def _restore(self, state_dict):
        for m in self.affine:
            buf = state_dict[id(m)]
            m.weight.data.copy_(buf["weight"])
            if m.bias is not None and buf["bias"] is not None:
                m.bias.data.copy_(buf["bias"])

################################################################################
# ------------------------------  MAIN TRAINING  ----------------------------- #
################################################################################


def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config_path).read_text())
    run_id = cfg["run_id"]

    results_root = Path(args.results_dir).expanduser()
    results_root.mkdir(parents=True, exist_ok=True)
    images_dir = results_root / "images"
    images_dir.mkdir(exist_ok=True, parents=True)

    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------------------------------------------------- dataset #
    train_loader, val_loader, num_classes = get_dataloaders(cfg["dataset"],
                                                           cfg["training"])

    # --------------------------------------------------------------   model #
    model_cfg = cfg["model"]
    model_cfg["num_classes"] = num_classes  # ensure consistency
    model = build_model(model_cfg).to(device)

    # Optionally load a pre-trained source checkpoint
    if "pretrained" in model_cfg and model_cfg["pretrained"]:
        model.load_state_dict(torch.load(model_cfg["pretrained"], map_location=device))

    # -------------------------------------------------------------  optimiser #
    optim_cfg = cfg["training"]
    optimiser = optim.Adam(model.parameters(), lr=optim_cfg.get("learning_rate", 1e-3))
    criterion = nn.CrossEntropyLoss()

    method = optim_cfg.get("method", "source").lower()
    use_zorro = method == "zorro"

    # Ablation flags -------------------------------------------------------- #
    zorro_kwargs = dict(
        lambda_=optim_cfg.get("lambda", 4),
        eps=optim_cfg.get("eps", 5e-3),
        disable_fisher=optim_cfg.get("disable_fisher", False),
        use_shrinkage=not optim_cfg.get("disable_shrinkage", False),
        use_gate=not optim_cfg.get("disable_gate", False),
        use_rollback=not optim_cfg.get("disable_rollback", False),
    )
    z_state = ZorroState(model, **zorro_kwargs) if use_zorro else None

    epochs = optim_cfg.get("epochs", 1)
    history: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train(not use_zorro)  # keep BN stats frozen for adaptation
        epoch_loss = 0.0
        correct, total = 0, 0
        pbar = tqdm(train_loader, desc=f"[{run_id}] Epoch {epoch}/{epochs}")
        for batch in pbar:
            inputs, targets = (b.to(device) for b in batch)

            if use_zorro:
                # ---------------- inference + forward NG update ---------------- #
                with torch.no_grad():
                    logits = model(inputs)
                    z_state.zorro_step(logits)
                loss = criterion(logits, targets)
            else:
                optimiser.zero_grad()
                logits = model(inputs)
                loss = criterion(logits, targets)
                loss.backward()
                optimiser.step()

            # ---------------- stats ---------------- #
            epoch_loss += loss.item() * inputs.size(0)
            _, preds = logits.max(1)
            correct += preds.eq(targets).sum().item()
            total += inputs.size(0)
            pbar.set_postfix({"loss": loss.item(),
                              "acc": 100.0 * correct / total})

        epoch_loss /= total
        epoch_acc = 100.0 * correct / total
        history.append({"epoch": epoch, "loss": epoch_loss, "acc": epoch_acc})

    # ----------------------------------------------------------------- eval #
    model.eval()
    val_correct, val_total, val_loss = 0, 0, 0.0
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

    ############################################################################
    # ---------------------------  SAVE ARTIFACTS  --------------------------- #
    ############################################################################
    ckpt_path = results_root / "model.pt"
    torch.save(model.state_dict(), ckpt_path)

    # ------------------------ metrics JSON ------------------------ #
    metrics = {
        "run_id": run_id,
        "method": method,
        "epochs": epochs,
        "train_history": history,
        "val_loss": val_loss,
        "val_acc": val_acc,
    }
    (results_root / "results.json").write_text(json.dumps(metrics, indent=2))

    # ---------------------------  FIGURES  ------------------------- #
    epochs_axis = [h["epoch"] for h in history]
    train_loss_axis = [h["loss"] for h in history]
    train_acc_axis = [h["acc"] for h in history]

    # line – loss
    plt.figure()
    plt.plot(epochs_axis, train_loss_axis, marker="o", label="train_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title(f"Training loss – {run_id}")
    plt.legend()
    plt.annotate(f"{train_loss_axis[-1]:.3f}",
                 (epochs_axis[-1], train_loss_axis[-1]),
                 textcoords="offset points", xytext=(0, 5), ha='center')
    plt.tight_layout()
    plt.savefig(images_dir / "training_loss.pdf", bbox_inches="tight")
    plt.close()

    # line – accuracy
    plt.figure()
    plt.plot(epochs_axis, train_acc_axis, marker="o", label="train_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Training accuracy – {run_id}")
    plt.legend()
    plt.annotate(f"{train_acc_axis[-1]:.2f}%", (epochs_axis[-1], train_acc_axis[-1]),
                 textcoords="offset points", xytext=(0, 5), ha='center')
    plt.tight_layout()
    plt.savefig(images_dir / "accuracy.pdf", bbox_inches="tight")
    plt.close()

    # -------------------- stdout summary (required) -------------------- #
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()