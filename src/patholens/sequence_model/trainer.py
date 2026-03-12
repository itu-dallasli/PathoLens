"""
Trainer — MIL training loop for the SlideEncoder.

Supports:
  * Mixed-precision training (torch AMP)
  * Gradient checkpointing
  * Cosine-annealing LR schedule with warmup
  * Slide-level classification with weak labels
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from patholens.logger import get_logger
from patholens.sequence_model.slide_encoder import SlideEncoder

log = get_logger(__name__)


class SlideTrainer:
    """
    Training loop for :class:`SlideEncoder`.

    Parameters
    ----------
    model : SlideEncoder
    config : object
        Should expose ``training.learning_rate``, ``training.weight_decay``,
        ``training.epochs``, ``training.warmup_epochs``,
        ``training.gradient_checkpointing``.
    device : str
    checkpoint_dir : str | Path
    """

    def __init__(
        self,
        model: SlideEncoder,
        config,
        device: str = "cuda:0",
        checkpoint_dir: str | Path = "checkpoints",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training config with safe attribute access
        tcfg = config.training if hasattr(config, "training") else config
        self.lr = getattr(tcfg, "learning_rate", 1e-4)
        self.wd = getattr(tcfg, "weight_decay", 1e-2)
        self.epochs = getattr(tcfg, "epochs", 50)
        self.warmup_epochs = getattr(tcfg, "warmup_epochs", 5)
        self.use_amp = getattr(config, "mixed_precision", True) if hasattr(config, "mixed_precision") else True
        self.use_grad_ckpt = getattr(tcfg, "gradient_checkpointing", True)

        # Optimiser
        self.optimizer = AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.wd
        )

        # LR schedule: linear warmup → cosine decay
        warmup = LinearLR(
            self.optimizer, start_factor=0.01, total_iters=self.warmup_epochs
        )
        cosine = CosineAnnealingLR(
            self.optimizer, T_max=self.epochs - self.warmup_epochs
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup, cosine],
            milestones=[self.warmup_epochs],
        )

        # Loss & AMP
        self.criterion = nn.CrossEntropyLoss()
        self.scaler = GradScaler(enabled=self.use_amp)

        # Gradient checkpointing
        if self.use_grad_ckpt:
            self._enable_gradient_checkpointing()

        log.info(
            "Trainer ready  |  lr=%.1e  wd=%.1e  epochs=%d  amp=%s  grad_ckpt=%s",
            self.lr,
            self.wd,
            self.epochs,
            self.use_amp,
            self.use_grad_ckpt,
        )

    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing on Mamba encoder layers."""
        for module in self.model.modules():
            if hasattr(module, "gradient_checkpointing_enable"):
                module.gradient_checkpointing_enable()

    # ── Training loop ────────────────────────────────────────
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> Dict[str, list]:
        """
        Run the full training loop.

        Parameters
        ----------
        train_loader
            Yields ``(patch_embeddings, label)`` per slide.
            ``patch_embeddings``: (1, N, 1024)
            ``label``: (1,) int64
        val_loader
            Optional validation loader.

        Returns
        -------
        Dict with ``train_loss``, ``val_loss``, ``val_acc`` history lists.
        """
        history: Dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "val_acc": [],
        }

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            train_loss = self._train_epoch(train_loader)
            history["train_loss"].append(train_loss)

            val_loss, val_acc = 0.0, 0.0
            if val_loader is not None:
                val_loss, val_acc = self._validate(val_loader)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            self.scheduler.step()
            elapsed = time.time() - t0

            log.info(
                "Epoch %3d/%d  |  train_loss=%.4f  val_loss=%.4f  "
                "val_acc=%.2f%%  lr=%.2e  time=%.1fs",
                epoch,
                self.epochs,
                train_loss,
                val_loss,
                val_acc * 100,
                self.optimizer.param_groups[0]["lr"],
                elapsed,
            )

            # Save checkpoint every 10 epochs
            if epoch % 10 == 0 or epoch == self.epochs:
                self._save_checkpoint(epoch, val_loss)

        return history

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        n = 0
        for embeddings, labels in loader:
            embeddings = embeddings.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.use_amp):
                output = self.model(embeddings)
                loss = self.criterion(output.classification_logits, labels)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            n += 1

        return total_loss / max(n, 1)

    @torch.no_grad()
    def _validate(self, loader: DataLoader):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        for embeddings, labels in loader:
            embeddings = embeddings.to(self.device)
            labels = labels.to(self.device)

            with autocast(enabled=self.use_amp):
                output = self.model(embeddings)
                loss = self.criterion(output.classification_logits, labels)

            total_loss += loss.item()
            preds = output.classification_logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.shape[0]

        avg_loss = total_loss / max(total, 1)
        accuracy = correct / max(total, 1)
        return avg_loss, accuracy

    # ── Checkpointing ────────────────────────────────────────
    def _save_checkpoint(self, epoch: int, val_loss: float):
        path = self.checkpoint_dir / f"slide_encoder_epoch{epoch:03d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "val_loss": val_loss,
            },
            path,
        )
        log.info("Checkpoint saved → %s", path)

    def load_checkpoint(self, path: str | Path):
        """Load a saved checkpoint to resume training."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        log.info("Resumed from %s (epoch %d)", path, ckpt.get("epoch", "?"))
