"""
Sentence embeddings with all-MiniLM-L6-v2 on ONNX Runtime (no PyTorch).

Same weights as sentence-transformers' all-MiniLM-L6-v2 (its official ONNX export), with the same mean pooling and
L2 normalisation, so vectors match the PyTorch model (cosine 1.0 on the eval queries). ONNX Runtime keeps the
deployed backend small enough for a Vercel Function.
"""
import threading
from functools import lru_cache

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from config import EMBED_MODEL_DIR

MAX_TOKENS = 256  # the model's max_seq_length
_lock = threading.Lock()


@lru_cache
def _session() -> tuple[ort.InferenceSession, Tokenizer]:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    session = ort.InferenceSession(str(EMBED_MODEL_DIR / "model.onnx"), opts, providers=["CPUExecutionProvider"])
    tok = Tokenizer.from_file(str(EMBED_MODEL_DIR / "tokenizer.json"))
    tok.enable_truncation(MAX_TOKENS)
    tok.enable_padding()
    return session, tok


def encode(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """(len(texts), 384) float32, L2-normalised."""
    session, tok = _session()
    inputs = {i.name for i in session.get_inputs()}
    out = []
    for i in range(0, len(texts), batch_size):
        with _lock:  # the tokenizer's padding state is shared
            enc = tok.encode_batch(texts[i:i + batch_size])
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        feeds = {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)}
        hidden = session.run(None, {k: v for k, v in feeds.items() if k in inputs})[0]
        pooled = (hidden * mask[..., None]).sum(1) / mask.sum(1, keepdims=True)
        out.append(pooled / np.linalg.norm(pooled, axis=1, keepdims=True))
    return np.concatenate(out).astype(np.float32)
