from __future__ import annotations

from typing import Iterable, List

import numpy as np
from sentence_transformers import SentenceTransformer


class LocalSentenceEmbedder:
    def __init__(self, model_path: str, device: str = "cpu", batch_size: int = 32, normalize: bool = True):
        self.model = SentenceTransformer(model_path, device=device)
        self.batch_size = batch_size
        self.normalize = normalize

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, 0), dtype=float)
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        return embeddings.astype(float)
