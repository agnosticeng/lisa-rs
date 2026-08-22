#!/usr/bin/env python3
"""Reference token embedding, final RMSNorm, and a 32-row lm_head slice."""
import glob
import json
import os
import struct

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import mlx_env as _mlx_env
_mlx_env.reexec()
import mlx.core as mx
import mlx.nn as nn
import numpy as np

HOME = os.path.expanduser("~")
CACHE = f"{HOME}/.cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-4bit/snapshots"
SHARDS = sorted(glob.glob(f"{CACHE}/*/*.safetensors"))
TOKEN = 2005
HIDDEN = 5120
HEAD_ROWS = 248320

def load(name):
    for path in SHARDS:
        with open(path, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_len).decode())
            if name not in header:
                continue
            start, end = header[name]["data_offsets"]
            f.seek(8 + header_len + start)
            raw = f.read(end - start)
        if header[name]["dtype"] == "U32":
            return np.frombuffer(raw, dtype="<u4").copy()
        bits = np.frombuffer(raw, dtype="<u2").copy()
        return (bits.astype(np.uint32) << 16).view(np.float32).copy()
    raise KeyError(name)


def load_rows(name, first, rows):
    for path in SHARDS:
        with open(path, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_len).decode())
            if name not in header:
                continue
            info = header[name]
            start, end = info["data_offsets"]
            row_bytes = (end - start) // info["shape"][0]
            f.seek(8 + header_len + start + first * row_bytes)
            raw = f.read(rows * row_bytes)
        if info["dtype"] == "U32":
            return np.frombuffer(raw, dtype="<u4").copy()
        bits = np.frombuffer(raw, dtype="<u2").copy()
        return (bits.astype(np.uint32) << 16).view(np.float32).copy()
    raise KeyError(name)


def quantized_embedding_row(prefix, row):
    embedding = nn.QuantizedEmbedding(1, HIDDEN, group_size=64, bits=4, mode="affine")
    embedding.weight = mx.array(load_rows(prefix + ".weight", row, 1).reshape(1, HIDDEN // 8))
    embedding.scales = mx.array(load_rows(prefix + ".scales", row, 1).reshape(1, HIDDEN // 64)).astype(mx.bfloat16)
    embedding.biases = mx.array(load_rows(prefix + ".biases", row, 1).reshape(1, HIDDEN // 64)).astype(mx.bfloat16)
    return embedding(mx.array([0], dtype=mx.uint32))


embed = quantized_embedding_row("language_model.model.embed_tokens", TOKEN)
norm = nn.RMSNorm(HIDDEN, eps=1e-6)
norm.weight = mx.array(load("language_model.model.norm.weight")).astype(mx.bfloat16)
normalized = norm(embed)

head = nn.QuantizedLinear(HIDDEN, HEAD_ROWS, bias=False, group_size=64, bits=4, mode="affine")
head.weight = mx.array(load_rows("language_model.lm_head.weight", 0, HEAD_ROWS).reshape(HEAD_ROWS, HIDDEN // 8))
head.scales = mx.array(load_rows("language_model.lm_head.scales", 0, HEAD_ROWS).reshape(HEAD_ROWS, HIDDEN // 64)).astype(mx.bfloat16)
head.biases = mx.array(load_rows("language_model.lm_head.biases", 0, HEAD_ROWS).reshape(HEAD_ROWS, HIDDEN // 64)).astype(mx.bfloat16)
logits = head(normalized)

np.asarray(embed.astype(mx.float32), dtype=np.float32).tofile("/tmp/mlx_io_embed.raw")
np.asarray(normalized.astype(mx.float32), dtype=np.float32).tofile("/tmp/mlx_io_norm.raw")
np.asarray(logits.astype(mx.float32), dtype=np.float32).tofile("/tmp/mlx_io_logits.raw")
print("saved embed/norm/logits", embed.shape, normalized.shape, logits.shape)
