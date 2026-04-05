"""
UNI Feature Extractor — Frozen ViT-L/16 patch embedding.

Uses the MahmoodLab/UNI model (ViT-Large, patch-16, 224px) to produce
1024-dimensional feature vectors from histopathology image patches.
The model weights are loaded once and kept frozen (no gradient computation).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from patholens.logger import get_logger

log = get_logger(__name__)


class UNIFeatureExtractor:
    """
    Frozen UNI patch-level feature extractor.

    Parameters
    ----------
    model_name : str
        HuggingFace repo id (default ``"MahmoodLab/UNI"``).
    device : str
        Torch device.
    normalize : bool
        Whether to L2-normalise output embeddings.
    """

    EMBEDDING_DIM = 1024

    def __init__(
        self,
        model_name: str = "MahmoodLab/UNI",
        device: str = "cuda:0",
        normalize: bool = True,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.normalize = normalize
        self.model_name = model_name

        self.model = self._load_model()
        self.transform = self._build_transform()

        log.info(
            "UNI feature extractor ready  |  device=%s  normalize=%s",
            self.device,
            self.normalize,
        )

    # ── Model loading ────────────────────────────────────────
    def _load_model(self) -> torch.nn.Module:
        """Load UNI weights from HuggingFace via timm."""
        try:
            import timm
            from huggingface_hub import hf_hub_download, login
        except ImportError as e:
            raise ImportError(
                "timm and huggingface_hub are required: "
                "pip install timm huggingface-hub"
            ) from e

        # Authenticate non-interactively: prefer HF_TOKEN env var, then cached token
        import os
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            cached = Path.home() / ".cache" / "huggingface" / "token"
            if cached.exists():
                hf_token = cached.read_text().strip()
        if hf_token:
            try:
                login(token=hf_token, add_to_git_credential=False)
            except Exception:
                log.warning("HuggingFace login failed. Model download may fail.")
        else:
            log.warning(
                "No HuggingFace token found. Set HF_TOKEN or run "
                "`huggingface-cli login`."
            )

        # Download model weights
        log.info("Downloading UNI weights from %s ...", self.model_name)
        ckpt_path = hf_hub_download(
            repo_id=self.model_name,
            filename="pytorch_model.bin",
        )

        # Create ViT-L/16 architecture (no classification head)
        model = timm.create_model(
            "vit_large_patch16_224",
            pretrained=False,
            init_values=1e-5,
            num_classes=0,  # remove classifier → feature-only
        )

        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=True)
        model = model.to(self.device)
        model.eval()

        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False

        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        log.info("UNI loaded: %.1fM parameters (frozen)", n_params)
        return model

    # ── Preprocessing ────────────────────────────────────────
    @staticmethod
    def _build_transform() -> transforms.Compose:
        """ImageNet-normalised transform for 224×224 input."""
        return transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    # ── Extraction ───────────────────────────────────────────
    @torch.no_grad()
    def extract_batch(self, images: List[Image.Image]) -> torch.Tensor:
        """
        Extract embeddings for a list of PIL images.

        Parameters
        ----------
        images : list[PIL.Image]
            RGB patch images (any size; will be resized to 224).

        Returns
        -------
        torch.Tensor
            Shape ``(N, 1024)`` float32 on CPU.
        """
        tensors = [self.transform(img) for img in images]
        batch = torch.stack(tensors).to(self.device)

        embeddings = self.model(batch)  # (N, 1024)

        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings.cpu()

    @torch.no_grad()
    def extract_single(self, image: Image.Image) -> np.ndarray:
        """Extract embedding for one image → (1024,) numpy array."""
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        emb = self.model(tensor)
        if self.normalize:
            emb = F.normalize(emb, p=2, dim=-1)
        return emb.cpu().numpy().squeeze(0)
