#!/usr/bin/env python
"""
Training / adaptation runner for a SINGLE experiment variation.
This file can be launched stand-alone for debugging but is normally
spawned by src.main so that every run is isolated and fully logged.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path
from typing import Dict, List

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

matplotlib.use("Agg")

################################################################################
#                               ARG PARSING                                    #
################################################################################

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config-path", type=str, required=True,
                   help="YAML file that describes *one* experiment run")
    p.add_argument("--results-dir", type=str, required=True,
                   help="Directory where this run stores all artefacts")
    return p.parse_args()

################################################################################
#                        ZORRO & NGFAT ADAPTATION STATE                        #
################################################################################

class ZorroState:
    """Implements ZORRO-full adaptation (gate + shrinkage + rollback)."""

    def __init__(self, model: nn.Module, lambda_: float = 8.0,
                 eps: float = 1e-2, k_ckpt: int = 3):
        self.lambda_ = lambda_
        self.eps = eps
        self.n = 0                              # effective sample count
        self.last_acc_hat = 1.0
        self.last_entropy = 0.0
        self.ckpts: deque = deque(maxlen=k_ckpt)
        self.affine: List[nn.Module] = []
        self.act_cache: Dict[int, torch.Tensor] = {}
        self._register_hooks(model)

    # ---------------------------------------------------------------------
    def _register_hooks(self, model):
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d,
                              nn.GroupNorm, nn.LayerNorm)):
                self.affine.append(m)
                m.register_forward_hook(self._make_hook(m))

    def _make_hook(self, module):
        def hook(_, __, output):
            self.act_cache[id(module)] = output.detach()
        return hook

    # ---------------------------------------------------------------------
    @torch.no_grad()
    def _snapshot(self):
        return {id(m): m.weight.data.clone() for m in self.affine}

    @torch.no_grad()
    def _restore(self, snap):
        for m in self.affine:
            m.weight.data.copy_(snap[id(m)])

    # ---------------------------------------------------------------------
    @torch.no_grad()
    def step(self, logits):
        prob = torch.softmax(logits, 1)
        logp = torch.log_softmax(logits, 1)
        entropy = -(prob * logp).sum(1)
        var_proxy = (prob * (1 - prob)).sum(1)
        acc_hat = 1 - var_proxy.mean()

        update = (acc_hat < self.last_acc_hat - self.eps) or \
                 (entropy.mean() > self.last_entropy * 0.9)
        tau = self.n / (self.n + self.lambda_) if (self.n + self.lambda_) > 0 else 0.0

        if update:
            for m in self.affine:
                y = self.act_cache[id(m)]
                if y.ndim > 2:
                    g = y.mean((0, 2, 3))
                    F_diag = y.var((0, 2, 3)) + 1e-5
                else:
                    g = y.mean(0)
                    F_diag = y.var(0) + 1e-5
                step = -tau * g / F_diag
                m.weight.data.add_(step.view_as(m.weight))
            self.ckpts.append((self._snapshot(), acc_hat.item()))
        else:
            if len(self.ckpts) == self.ckpts.maxlen and \
               acc_hat.item() > max(a for _, a in self.ckpts):
                snap, _ = max(self.ckpts, key=lambda x: x[1])
                self._restore(snap)

        self.last_acc_hat = acc_hat.item()
        self.last_entropy = entropy.mean().item()
        self.n += 1

################################################################################
#                          TRAINING / ADAPTATION LOOP                          #
################################################################################

def freeze_all(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = False

def collect_affine_params(model: nn.Module):
    params = []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d,
                          nn.GroupNorm, nn.LayerNorm)):
            if m.weight is not None:
                params.append(m.weight)
            if m.bias is not None:
                params.append(m.bias)
    return params


@torch.no_grad()
def bn_adapt_forward(model: nn.Module, x: torch.Tensor):
    """Just a forward pass in training mode so running stats update."""
    model.train()
    return model(x)

################################################################################
#                                    MAIN                                      #
################################################################################

def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config_path).read_text())
    run_id: str = cfg["run_id"]

    results_root = Path(args.results_dir).expanduser()
    results_root.mkdir(parents=True, exist_ok=True)
    images_dir = results_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    set_seed(cfg.get("seed", 0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------------------------------------------------- dataset
    train_loader, val_loader, num_classes = get_dataloaders(cfg["dataset"],
                                                           cfg["training"])

    # ---------------------------------------------------------------- model
    model_cfg = cfg["model"]
    model_cfg["num_classes"] = num_classes
    model = build_model(model_cfg).to(device)

    # optional pretrained
    if model_cfg.get("pretrained"):
        model.load_state_dict(torch.load(model_cfg["pretrained"], map_location=device))

    # ---------------------------------------------------------------- method
    method = cfg["training"].get("method", "source").lower()
    epochs = cfg["training"].get("epochs", 1)
    lr = cfg["training"].get("learning_rate", 1e-3)

    # generic optimiser for supervised training
    optimiser = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # specialised states / optimisers
    if method == "tent":
        freeze_all(model)
        for p in collect_affine_params(model):
            p.requires_grad = True
        tent_opt = optim.SGD(collect_affine_params(model), lr=lr)
    elif method == "bn_adapt":
        freeze_all(model)
    elif method == "source_frozen":
        freeze_all(model)
    elif method == "ngfat":
        freeze_all(model)
        ngf_affine = collect_affine_params(model)
    elif "zorro" in method:
        freeze_all(model)
        z_state = ZorroState(model,
                             lambda_=cfg["training"].get("lambda", 8.0),
                             eps=cfg["training"].get("eps", 1e-2))
    # ---------------------------------------------------------------- loop
    history: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        epoch_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"[{run_id}] Epoch {epoch}/{epochs}")

        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            if method == "source":
                model.train()
                optimiser.zero_grad()
                logits = model(inputs)
                loss = criterion(logits, targets)
                loss.backward()
                optimiser.step()

            elif method == "source_frozen":
                model.eval()
                logits = model(inputs)
                loss = criterion(logits, targets)

            elif method == "bn_adapt":
                logits = bn_adapt_forward(model, inputs)
                loss = criterion(logits, targets)

            elif method == "tent":
                model.train()
                tent_opt.zero_grad()
                logits = model(inputs)
                prob = torch.softmax(logits, 1)
                ent_loss = (-(prob * torch.log(prob + 1e-6)).sum(1)).mean()
                ent_loss.backward()
                tent_opt.step()
                loss = criterion(logits.detach(), targets)

            elif method == "ngfat":
                model.eval()
                with torch.no_grad():
                    logits = model(inputs)
                    prob = torch.softmax(logits, 1)
                    logp = torch.log_softmax(logits, 1)
                    ent = -(prob * logp).sum(1)
                    g = prob - prob.mean(0, keepdim=True)  # rough grad proxy
                    # simple natural-gradient diag approximation
                    for m in model.modules():
                        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d,
                                          nn.GroupNorm, nn.LayerNorm)) and m.weight.requires_grad:
                            y = m.weight.data
                            step = -0.001 * g.mean()  # tiny LR
                            m.weight.data.add_(step)
                loss = criterion(logits, targets)

            else:  # zorro variants
                model.eval()
                logits = model(inputs)
                z_state.step(logits)
                loss = criterion(logits, targets)

            epoch_loss += loss.item() * inputs.size(0)
            _, preds = logits.max(1)
            correct += preds.eq(targets).sum().item()
            total += inputs.size(0)
            pbar.set_postfix({"loss": loss.item(), "acc": 100. * correct / total})

        history.append({"epoch": epoch,
                        "loss": epoch_loss / total,
                        "acc": 100. * correct / total})

    # ---------------------------------------------------------------- eval
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            val_loss += criterion(out, y).item() * x.size(0)
            val_correct += out.argmax(1).eq(y).sum().item()
            val_total += x.size(0)
    val_loss /= val_total
    val_acc = 100. * val_correct / val_total

    ################################################################################
    #                                SAVE ARTEFACTS                               #
    ################################################################################
    torch.save(model.state_dict(), results_root / "model.pt")

    metrics = {"run_id": run_id,
               "method": method,
               "train_history": history,
               "val_loss": val_loss,
               "val_acc": val_acc}
    (results_root / "results.json").write_text(json.dumps(metrics, indent=2))

    # --------------- figures
    epochs_axis = [h["epoch"] for h in history]
    loss_axis = [h["loss"] for h in history]
    acc_axis = [h["acc"] for h in history]

    plt.figure(); plt.plot(epochs_axis, loss_axis, marker="o");
    plt.title(f"Train loss – {run_id}"); plt.xlabel("Epoch"); plt.ylabel("CE loss");
    plt.tight_layout(); plt.savefig(images_dir / "training_loss.pdf"); plt.close()

    plt.figure(); plt.plot(epochs_axis, acc_axis, marker="o");
    plt.title(f"Train acc – {run_id}"); plt.xlabel("Epoch"); plt.ylabel("Accuracy %");
    plt.tight_layout(); plt.savefig(images_dir / "accuracy.pdf"); plt.close()

    # stdout summary for orchestrator
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()