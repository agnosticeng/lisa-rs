#!/usr/bin/env python3
"""Write full-model Qwen3.8-27B logits for the one-token input [2005]."""

import glob
import os

import mlx.core as mx
import numpy as np
from mlx_vlm import load


ROOT = os.path.expanduser(
    "~/.cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-4bit/snapshots"
)
snapshots = sorted(
    path
    for path in glob.glob(f"{ROOT}/*")
    if glob.glob(f"{path}/*.safetensors")
)
if not snapshots:
    raise SystemExit(f"no cached snapshot below {ROOT}")

snapshot = snapshots[-1]
model, _processor = load(snapshot)
token = mx.array([[2005]], dtype=mx.int32)
output = model.language_model(token)
logits = output.logits[0, -1].astype(mx.float32)
mx.eval(logits)
values = np.asarray(logits, dtype=np.float32)
if values.shape != (248320,):
    raise RuntimeError(f"unexpected logits shape {values.shape}")
values.tofile("/tmp/mlx_full_token_2005.raw")
print(
    "saved /tmp/mlx_full_token_2005.raw",
    values.shape,
    "argmax",
    int(values.argmax()),
)
