#!/usr/bin/env python3
"""TRUE independent reference for full-attention layer 3 on real Qwen3.8-27B-4bit,
using MLX's OWN quantized matmul + the ACTUAL mlx-vlm Qwen3_5Attention module
(unmodified). Feeds the same deterministic bf16 h as tests/attn_decode.rs and
dumps the attention output (o_proj result) to /tmp/mlx_attn_truth.raw, so the
Rust engine can be anchored to ground truth (not to a re-implementation).

Run: python3 scripts/mlx_attn_truth.py
"""
import os, struct, json, glob, mmap
import numpy as np
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import mlx_env as _mlx_env
_mlx_env.reexec()
import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.models.qwen3_5.language import Qwen3_5Attention

HOME = os.path.expanduser("~")
CACHE = f"{HOME}/.cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-4bit/snapshots"
shards = sorted(glob.glob(f"{CACHE}/*/*.safetensors"))
M, H = 16, 5120
LAYER = "language_model.model.layers.3.self_attn."


def read_tensor(path, name):
    with open(path, "rb") as f:
        data = f.read()
    n = struct.unpack("<Q", data[:8])[0]
    header = json.loads(data[8:8 + n].decode())
    data_off = 8 + n
    s, e = header[name]["data_offsets"]
    return data[data_off + s:data_off + e], header[name]["dtype"]


def load(path, name):
    raw, dt = read_tensor(path, name)
    if dt == "U32":
        return np.frombuffer(raw, dtype="<u4").reshape(-1).copy()
    if dt == "BF16":
        b = np.frombuffer(raw, dtype="<u2").copy()
        return (b.astype(np.uint32) << 16).view(np.float32).copy()
    raise ValueError(dt)


def find_shard(tname):
    for p in shards:
        with open(p, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n).decode())
            if tname in header:
                return p
    raise SystemExit(f"{tname} not found")


shard = find_shard(LAYER + "q_proj.weight")


def ql(proj, out, inn):
    w = load(shard, LAYER + proj + ".weight").reshape(out, inn // 8)
    s = load(shard, LAYER + proj + ".scales").reshape(out, inn // 64)
    b = load(shard, LAYER + proj + ".biases").reshape(out, inn // 64)
    lin = nn.QuantizedLinear(inn, out, bias=False, group_size=64, bits=4, mode="affine")
    lin.weight = mx.array(w.astype(np.uint32))
    lin.scales = mx.array(s)
    lin.biases = mx.array(b)
    return lin


from types import SimpleNamespace

args = SimpleNamespace(
    num_key_value_heads=4,
    num_attention_heads=24,
    head_dim=256,
    hidden_size=H,
    attention_bias=False,
    rms_norm_eps=1e-6,
    max_position_embeddings=262144,
    rope_parameters={"mrope_interleaved": True, "mrope_section": [11, 11, 10],
                     "partial_rotary_factor": 0.25, "rope_theta": 1e7,
                     "rope_type": "default"},
)

attn = Qwen3_5Attention(args)
attn.q_proj = ql("q_proj", 12288, H)
attn.k_proj = ql("k_proj", 1024, H)
attn.v_proj = ql("v_proj", 1024, H)
attn.o_proj = ql("o_proj", 5120, 6144)
qn = load(shard, LAYER + "q_norm.weight").reshape(256)
kn = load(shard, LAYER + "k_norm.weight").reshape(256)
attn.q_norm.weight = mx.array(qn.astype(np.float32))
attn.k_norm.weight = mx.array(kn.astype(np.float32))

# deterministic hidden, same as tests/attn_decode.rs
hvals = np.array([(i % 97) * 0.001 for i in range(M * H)], dtype=np.float32)
h = mx.array(hvals, mx.float32).reshape(1, M, H).astype(mx.bfloat16)

# position ids: [B, T] world == text-only; Qwen3_5Attention tiles to 3 axes
pos = mx.arange(M, dtype=mx.int32)[None, :]

out = attn(h, mask="causal", cache=None, position_ids=pos)
res = np.asarray(out, dtype=np.float32).reshape(M, H)
res.astype(np.float32).tofile("/tmp/mlx_attn_truth.raw")
np.save("/tmp/mlx_attn_truth.npy", res)
print(f"truth attn o_out[0,:4] = {res[0,0]:.6f} {res[0,1]:.6f} {res[0,2]:.6f} {res[0,3]:.6f}")
print("saved /tmp/mlx_attn_truth.raw", res.shape)

# Capture the pre-o_proj gated context by re-running with a recording o_proj.
import mlx.nn as nn


class Recorder(nn.QuantizedLinear):
    captured = None

    def __call__(self, x):
        Recorder.captured = x
        return x  # ignore o_proj on this pass

cap = attn.o_proj
attn.o_proj = Recorder(6144, 5120, bias=False, group_size=64, bits=4, mode="affine")
attn.o_proj.weight = cap.weight
attn.o_proj.scales = cap.scales
attn.o_proj.biases = cap.biases
attn(h, mask="causal", cache=None, position_ids=pos)
gated_np = np.asarray(Recorder.captured, dtype=np.float32).reshape(M, -1)
gated_np.astype(np.float32).tofile("/tmp/mlx_attn_gated_truth.raw")
print(f"truth gated[0,:4] = {gated_np[0,0]:.6f} {gated_np[0,1]:.6f} {gated_np[0,2]:.6f} {gated_np[0,3]:.6f}")
