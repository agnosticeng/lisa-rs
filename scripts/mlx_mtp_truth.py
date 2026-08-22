#!/usr/bin/env python3
"""Write deterministic M=1 Qwen3.8 MTP hidden state and logits."""

import glob
import os

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import mlx_env as _mlx_env
_mlx_env.reexec()
import mlx.core as mx
import numpy as np
from mlx_vlm import load
from mlx_vlm.speculative.drafters import load_drafter


def snapshot(model: str) -> str:
    root = os.path.expanduser(
        f"~/.cache/huggingface/hub/models--mlx-community--{model}/snapshots"
    )
    snapshots = sorted(
        path
        for path in glob.glob(f"{root}/*")
        if glob.glob(f"{path}/*.safetensors")
    )
    if not snapshots:
        raise SystemExit(f"no cached snapshot below {root}")
    return snapshots[-1]


target, _processor = load(snapshot("Qwen3.8-27B-4bit"))
draft, kind = load_drafter(snapshot("Qwen3.8-27B-MTP-4bit"))
if kind != "mtp":
    raise RuntimeError(f"unexpected drafter kind {kind}")
draft.bind(target)

token = mx.array([[2005]], dtype=mx.int32)
values = (np.arange(5120, dtype=np.int32) % 257 - 128).astype(np.float32) / 64.0
target_hidden = mx.array(values).astype(mx.bfloat16).reshape(1, 1, 5120)
token_embed = draft._input_embed(token)
cache = draft.make_cache()
hidden = draft._forward_hidden(
    token_embed,
    target_hidden,
    cache,
    mx.array([[0]], dtype=mx.int32),
)
logits = draft._lm_head_fn(hidden)[0, 0].astype(mx.float32)
hidden = hidden[0, 0].astype(mx.float32)
mx.eval(hidden, logits)

hidden_values = np.asarray(hidden, dtype=np.float32)
logit_values = np.asarray(logits, dtype=np.float32)
hidden_values.tofile("/tmp/mlx_mtp_hidden.raw")
logit_values.tofile("/tmp/mlx_mtp_logits.raw")
print(
    "saved /tmp/mlx_mtp_{hidden,logits}.raw",
    "argmax",
    int(logit_values.argmax()),
)
