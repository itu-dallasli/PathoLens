"""
Batch Processor — Efficient batch embedding extraction for entire WSIs.

Reads patches on-the-fly from coordinates stored in HDF5, sends them
through the UNI extractor in configurable batches, and writes the
resulting embeddings to disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from patholens.embedding.uni_extractor import UNIFeatureExtractor
from patholens.embedding.embedding_store import EmbeddingStore
from patholens.preprocessing.wsi_reader import WSIReader
from patholens.logger import get_logger

log = get_logger(__name__)


class BatchEmbeddingProcessor:
    """
    Process all patches of a WSI through the UNI extractor.

    Parameters
    ----------
    extractor : UNIFeatureExtractor
        Pre-loaded frozen feature extractor.
    store : EmbeddingStore
        Where to persist the resulting embeddings.
    batch_size : int
        Number of patches per forward pass.
    """

    def __init__(
        self,
        extractor: UNIFeatureExtractor,
        store: EmbeddingStore,
        batch_size: int = 256,
    ):
        self.extractor = extractor
        self.store = store
        self.batch_size = batch_size

    def process_slide(
        self,
        slide_id: str,
        wsi_path: str | Path,
        coordinates: np.ndarray,
        patch_size: int = 256,
        level: int = 0,
        skip_existing: bool = True,
    ) -> np.ndarray:
        """
        Embed every patch of a slide.

        Parameters
        ----------
        slide_id : str
            Unique identifier for the slide.
        wsi_path : str | Path
            Path to the WSI file.
        coordinates : np.ndarray
            (N, 2) array of level-0 (x, y) coordinates.
        patch_size : int
            Side of each square patch.
        level : int
            Pyramid level to read from.
        skip_existing : bool
            If True and embeddings already exist, skip.

        Returns
        -------
        np.ndarray
            (N, 1024) float32 embeddings.
        """
        if skip_existing and self.store.exists(slide_id):
            log.info("Embeddings already exist for %s — skipping.", slide_id)
            emb, _ = self.store.load(slide_id)
            return emb

        n_patches = len(coordinates)
        log.info(
            "Embedding slide %s  |  %d patches  |  batch_size=%d",
            slide_id,
            n_patches,
            self.batch_size,
        )

        all_embeddings: list[torch.Tensor] = []

        with WSIReader(wsi_path) as reader:
            for start in tqdm(
                range(0, n_patches, self.batch_size),
                desc=f"Embedding {slide_id}",
                unit="batch",
            ):
                end = min(start + self.batch_size, n_patches)
                batch_coords = coordinates[start:end]

                # Read patch images
                images: list[Image.Image] = []
                for x, y in batch_coords:
                    img = reader.read_region(
                        location=(int(x), int(y)),
                        level=level,
                        size=(patch_size, patch_size),
                    )
                    images.append(img)

                # Extract embeddings
                batch_emb = self.extractor.extract_batch(images)  # (B, 1024) CPU
                all_embeddings.append(batch_emb)

        embeddings = torch.cat(all_embeddings, dim=0).numpy()  # (N, 1024)

        # Persist
        self.store.save(
            slide_id=slide_id,
            embeddings=embeddings,
            coordinates=coordinates,
            attrs={
                "slide_id": slide_id,
                "wsi_path": str(wsi_path),
                "patch_size": patch_size,
                "level": level,
                "embedding_dim": UNIFeatureExtractor.EMBEDDING_DIM,
            },
        )

        log.info(
            "Finished %s  |  embeddings shape=%s",
            slide_id,
            embeddings.shape,
        )
        return embeddings
