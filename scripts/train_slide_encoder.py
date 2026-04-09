"""
Train the hierarchical :class:`SlideEncoder` (patch Mamba -> region Mamba ->
attention pool -> classifier) on the cached UNI embeddings produced by
``scripts/build_embedding_cache.py``.

Pipeline
--------
1. Load every ``<slide_id>.npz`` under ``--cache-dir``.
2. Stratified 80/20 train/val split by label.
3. Wrap in a ``Dataset`` that yields ``(embeddings, label)`` per slide.
   Because slides have variable patch counts and the model already handles
   arbitrary sequence lengths, we use ``batch_size=1``.
4. Build a ``SlideEncoder`` with ``n_classes=2`` (normal vs tumor), train
   with :class:`SlideTrainer` (CE loss + cosine LR + optional AMP).
5. Save the best-val checkpoint to ``checkpoints/slide_encoder_best.pt``.

Usage
-----
    python scripts/train_slide_encoder.py \\
        --cache-dir data/processed/slide_cache \\
        --epochs 20 \\
        --d-model 256 \\
        --n-layers 4

The script does NOT touch any WSI — only the cached NPZ files.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from patholens.logger import get_logger
from patholens.sequence_model.slide_encoder import SlideEncoder
from patholens.sequence_model.trainer import SlideTrainer

log = get_logger(__name__)


# ── Dataset ──────────────────────────────────────────────────

@dataclass
class SlideSample:
    path: Path
    slide_id: str
    label: int
    n_patches: int


class SlideCacheDataset(Dataset):
    """
    Yields ``(patch_embeddings, label)`` per cached slide.

    ``patch_embeddings``: ``torch.float32`` of shape ``(N, 1024)``
    ``label``            : ``torch.long`` scalar
    """

    def __init__(self, samples: List[SlideSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s = self.samples[idx]
        data = np.load(s.path)
        emb = torch.from_numpy(data["embeddings"].astype(np.float32))  # (N, 1024)
        label = torch.tensor(int(data["label"]), dtype=torch.long)
        return emb, label


def _collate_single(batch):
    """Batch of 1 -> ``(1, N, 1024), (1,)``. Variable N across slides."""
    emb, label = batch[0]
    return emb.unsqueeze(0), label.unsqueeze(0)


# ── Discovery + split ────────────────────────────────────────

def discover_cache(cache_dir: Path) -> List[SlideSample]:
    samples: List[SlideSample] = []
    for p in sorted(cache_dir.glob("*.npz")):
        try:
            with np.load(p) as data:
                label = int(data["label"])
                n = int(data["embeddings"].shape[0])
        except Exception as e:
            log.warning("skipping %s: %s", p.name, e)
            continue
        samples.append(
            SlideSample(path=p, slide_id=p.stem, label=label, n_patches=n)
        )
    return samples


def stratified_split(
    samples: List[SlideSample], val_frac: float, seed: int
) -> Tuple[List[SlideSample], List[SlideSample]]:
    rng = random.Random(seed)
    by_label: dict[int, List[SlideSample]] = {}
    for s in samples:
        by_label.setdefault(s.label, []).append(s)

    train: List[SlideSample] = []
    val: List[SlideSample] = []
    for label, group in by_label.items():
        group = list(group)
        rng.shuffle(group)
        n_val = max(1, int(round(len(group) * val_frac))) if len(group) > 1 else 0
        val.extend(group[:n_val])
        train.extend(group[n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


# ── Main ─────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("data/processed/slide_cache"),
        help="Directory containing <slide_id>.npz files",
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=Path("checkpoints"),
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--region-size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--backbone", type=str, default="mamba",
        choices=["mamba", "attention"],
        help="Sequence mixing backbone: 'mamba' (SSM, O(N)) or 'attention' (MHSA, O(N^2))",
    )
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--no-amp", action="store_true",
        help="Disable mixed-precision (auto-disabled on CPU regardless)",
    )
    parser.add_argument(
        "--no-grad-ckpt", action="store_true",
        help="Disable gradient checkpointing",
    )
    args = parser.parse_args()

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── Discover cache ───────────────────────────────────────
    if not args.cache_dir.exists():
        log.error("cache dir not found: %s", args.cache_dir)
        log.error("run scripts/build_embedding_cache.py first")
        return 1

    samples = discover_cache(args.cache_dir)
    if not samples:
        log.error("no .npz cache files found in %s", args.cache_dir)
        return 1

    n_tumor = sum(1 for s in samples if s.label == 1)
    n_normal = sum(1 for s in samples if s.label == 0)
    total_patches = sum(s.n_patches for s in samples)

    print(f"Cache: {args.cache_dir}")
    print(f"  slides: {len(samples)}  (tumor={n_tumor}  normal={n_normal})")
    print(f"  total cached patches: {total_patches:,}")
    print(
        f"  avg patches/slide: {total_patches / max(len(samples), 1):.0f}"
    )

    train_samples, val_samples = stratified_split(
        samples, args.val_frac, args.seed
    )
    print(
        f"Split: train={len(train_samples)}  val={len(val_samples)}  "
        f"(val_frac={args.val_frac})"
    )

    train_loader = DataLoader(
        SlideCacheDataset(train_samples),
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=_collate_single,
    )
    val_loader = DataLoader(
        SlideCacheDataset(val_samples),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=_collate_single,
    ) if val_samples else None

    # ── Model ────────────────────────────────────────────────
    model = SlideEncoder(
        input_dim=1024,
        d_model=args.d_model,
        n_layers=args.n_layers,
        region_size=args.region_size,
        n_classes=2,
        dropout=args.dropout,
        backbone=args.backbone,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"Model: SlideEncoder(backbone={args.backbone}, d_model={args.d_model}, "
        f"n_layers={args.n_layers}, region_size={args.region_size})"
    )
    print(f"  parameters: {n_params/1e6:.2f}M")

    # ── Trainer ──────────────────────────────────────────────
    # SlideTrainer reads lr/wd/epochs/... from config.training and
    # mixed_precision from config.mixed_precision. Build a lightweight
    # SimpleNamespace so we don't need a full Config object.
    use_amp = (not args.no_amp) and args.device.startswith("cuda")
    cfg = SimpleNamespace(
        training=SimpleNamespace(
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            warmup_epochs=args.warmup_epochs,
            gradient_checkpointing=not args.no_grad_ckpt,
        ),
        mixed_precision=use_amp,
    )

    trainer = SlideTrainer(
        model=model,
        config=cfg,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
    )

    # ── Train ────────────────────────────────────────────────
    t0 = time.time()
    history = trainer.train(train_loader, val_loader=val_loader)
    elapsed = time.time() - t0
    print()
    print("=" * 60)
    print(f"Training done in {elapsed/60:.1f} min")

    # Best-val checkpoint (lowest val_loss)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if history["val_loss"] and val_loader is not None:
        best_epoch = int(np.argmin(history["val_loss"])) + 1
        best_val = history["val_loss"][best_epoch - 1]
        best_acc = history["val_acc"][best_epoch - 1]
        print(
            f"Best val: epoch={best_epoch}  loss={best_val:.4f}  "
            f"acc={best_acc*100:.1f}%"
        )
    else:
        best_epoch = args.epochs
        best_val = float("nan")
        best_acc = float("nan")

    # Always write a convenience copy of the final model weights
    final_path = args.checkpoint_dir / "slide_encoder_final.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "input_dim": 1024,
                "d_model": args.d_model,
                "n_layers": args.n_layers,
                "region_size": args.region_size,
                "n_classes": 2,
                "dropout": args.dropout,
                "backbone": args.backbone,
            },
            "history": history,
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
            "best_val_acc": best_acc,
        },
        final_path,
    )
    print(f"Saved final weights -> {final_path}")

    # Dump history as JSON for easy inspection
    hist_path = args.checkpoint_dir / "slide_encoder_history.json"
    hist_path.write_text(
        json.dumps(
            {
                "train_loss": history["train_loss"],
                "val_loss": history["val_loss"],
                "val_acc": history["val_acc"],
                "best_epoch": best_epoch,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved history    -> {hist_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
