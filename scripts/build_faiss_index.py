"""
Build FAISS retrieval index from trained SlideEncoder embeddings.

Loads every cached NPZ produced by ``build_embedding_cache.py``, runs the
trained ``SlideEncoder`` to produce a 256-dim slide representation, then
builds a flat cosine-similarity FAISS index over all training slides.

The resulting index is the backbone of PathoLens visual-similarity retrieval
(CMEA step): at inference time a new slide's representation is compared
against this index to find the top-k most visually similar training cases,
which are then cited in the FHIR report.

Usage
-----
    python scripts/build_faiss_index.py \\
        --cache-dir  data/processed/slide_cache \\
        --checkpoint checkpoints/slide_encoder_final.pt \\
        --out        data/faiss_index

Outputs
-------
    data/faiss_index/slide_index.faiss   FAISS flat IP index (cosine after L2 norm)
    data/faiss_index/metadata.json       {slide_id -> {label, int_id, split}}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

try:
    import faiss
except ImportError:
    print("ERROR: faiss not installed. Run: pip install faiss-cpu")
    sys.exit(1)

from patholens.logger import get_logger
from patholens.sequence_model.slide_encoder import SlideEncoder

log = get_logger(__name__)


# ── Helpers ───────────────────────────────────────────────────

def load_encoder(checkpoint_path: Path) -> SlideEncoder:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    model = SlideEncoder(
        input_dim=cfg.get("input_dim", 1024),
        d_model=cfg.get("d_model", 256),
        n_layers=cfg.get("n_layers", 4),
        region_size=cfg.get("region_size", 64),
        n_classes=cfg.get("n_classes", 2),
        dropout=0.0,              # no dropout at index time
        backbone=cfg.get("backbone", "mamba"),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    d_model = cfg.get("d_model", 256)
    log.info(
        "Loaded SlideEncoder from %s  (backbone=%s  d_model=%d)",
        checkpoint_path,
        cfg.get("backbone", "mamba"),
        d_model,
    )
    return model, d_model


@torch.no_grad()
def embed_slide(model: SlideEncoder, npz_path: Path) -> np.ndarray:
    """Run one cached slide through the encoder -> (d_model,) float32."""
    data = np.load(npz_path)
    emb = torch.from_numpy(data["embeddings"].astype(np.float32)).unsqueeze(0)  # (1, N, 1024)
    out = model(emb)
    return out.slide_repr[0].numpy().astype(np.float32)  # (d_model,)


# ── Main ──────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir", type=Path,
        default=Path("data/processed/slide_cache"),
        help="Directory containing <slide_id>.npz files",
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("checkpoints/slide_encoder_final.pt"),
        help="Trained SlideEncoder checkpoint",
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("data/faiss_index"),
        help="Output directory for index + metadata",
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Default top-k for retrieval (stored in metadata)",
    )
    args = parser.parse_args()

    # ── Validate inputs ──────────────────────────────────────
    if not args.cache_dir.exists():
        log.error("cache dir not found: %s", args.cache_dir)
        return 1
    if not args.checkpoint.exists():
        log.error("checkpoint not found: %s", args.checkpoint)
        log.error("run scripts/train_slide_encoder.py first")
        return 1

    npz_files = sorted(args.cache_dir.glob("*.npz"))
    if not npz_files:
        log.error("no .npz files in %s", args.cache_dir)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)

    # ── Load encoder ─────────────────────────────────────────
    model, d_model = load_encoder(args.checkpoint)

    # ── Embed all slides ──────────────────────────────────────
    vectors = []
    metadata: dict[str, dict] = {}

    print(f"Embedding {len(npz_files)} slides ...")
    for int_id, npz_path in enumerate(npz_files):
        slide_id = npz_path.stem
        try:
            data_meta = np.load(npz_path)
            label = int(data_meta["label"])
            vec = embed_slide(model, npz_path)
            vectors.append(vec)
            metadata[slide_id] = {
                "int_id": int_id,
                "label": label,
                "label_name": "tumor" if label == 1 else "normal",
                "npz_path": str(npz_path),
            }
            if (int_id + 1) % 10 == 0 or (int_id + 1) == len(npz_files):
                print(f"  {int_id + 1}/{len(npz_files)}  {slide_id}  label={label}")
        except Exception as e:
            log.warning("skipping %s: %s", slide_id, e)

    if not vectors:
        log.error("no slides embedded successfully")
        return 1

    # ── Build FAISS flat IP index (cosine after L2 normalisation) ──
    mat = np.stack(vectors, axis=0).astype(np.float32)   # (N, d_model)
    faiss.normalize_L2(mat)

    index = faiss.IndexFlatIP(d_model)   # inner product == cosine after normalisation
    index.add(mat)

    # int_id -> slide_id reverse map (stored in metadata already; write separate list too)
    id_map = {v["int_id"]: k for k, v in metadata.items()}

    # ── Save ─────────────────────────────────────────────────
    index_path = args.out / "slide_index.faiss"
    faiss.write_index(index, str(index_path))

    meta_out = {
        "d_model": d_model,
        "n_slides": len(vectors),
        "top_k": args.top_k,
        "slides": metadata,
        "id_map": {str(k): v for k, v in id_map.items()},   # JSON keys must be str
    }
    meta_path = args.out / "metadata.json"
    meta_path.write_text(json.dumps(meta_out, indent=2), encoding="utf-8")

    n_tumor  = sum(1 for v in metadata.values() if v["label"] == 1)
    n_normal = sum(1 for v in metadata.values() if v["label"] == 0)
    print()
    print("Index built:")
    print(f"  slides : {len(vectors)}  (tumor={n_tumor}  normal={n_normal})")
    print(f"  dim    : {d_model}")
    print(f"  index  -> {index_path}")
    print(f"  meta   -> {meta_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
