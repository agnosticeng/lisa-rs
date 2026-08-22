#!/usr/bin/env python3
"""TRUE independent reference for a FULL Qwen3.8 transformer block on real
27B-4bit weights, using MLX's OWN quantized matmul + the ACTUAL unmodified
mlx-vlm Qwen3_5ModelBlock module.

Layer 17 is a linear (GatedDeltaNet) block:
    h = input_layernorm(x) -> linear_attn -> r ;  h = x + r
    h = h + mlp(post_attention_layernorm(h))

Feeds the same deterministic bf16 x as tests/fwd_decode.rs and dumps the final
block output so the Rust engine can be anchored to ground truth.

Run: python3 scripts/mlx_fwd_truth.py
"""
import os, struct, json, glob
import numpy as np
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import mlx_env as _mlx_env
_mlx_env.reexec()
import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.models.qwen3_5.language import Qwen3_5DecoderLayer

HOME = os.path.expanduser("~")
CACHE = f"{HOME}/.cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-4bit/snapshots"
shards = sorted(glob.glob(f"{CACHE}/*/*.safetensors"))
M, H = 16, 5120
LAYER = "language_model.model.layers.17."


def find_shard(tname):
    for p in shards:
        with open(p, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n).decode())
            if tname in header:
                return p
    raise SystemExit(f"{tname} not found")


_buf = {}


def load(name):
    p = find_shard(name)
    if p not in _buf:
        with open(p, "rb") as f:
            _buf[p] = f.read()
    data = _buf[p]
    th = data[:8]
    n = struct.unpack("<Q", th)[0]
    header = json.loads(data[8:8 + n].decode())
    data_off = 8 + n
    s, e = header[name]["data_offsets"]
    raw = data[data_off + s:data_off + e]
    dt = header[name]["dtype"]
    if dt == "U32":
        return np.frombuffer(raw, dtype="<u4").reshape(-1).copy()
    if dt == "BF16":
        b = np.frombuffer(raw, dtype="<u2").copy()
        return (b.astype(np.uint32) << 16).view(np.float32).copy()
    raise ValueError(dt)


def ql(proj, out, inn):
    w = load(proj + ".weight").reshape(out, inn // 8)
    s = load(proj + ".scales").reshape(out, inn // 64)
    b = load(proj + ".biases").reshape(out, inn // 64)
    lin = nn.QuantizedLinear(inn, out, bias=False, group_size=64, bits=4, mode="affine")
    lin.weight = mx.array(w.astype(np.uint32))
    lin.scales = mx.array(s).astype(mx.bfloat16)
    lin.biases = mx.array(b).astype(mx.bfloat16)
    return lin


from types import SimpleNamespace

# GDN layer-17 config (is_linear=True: (17+1)%4 != 0)
args = SimpleNamespace(
    hidden_size=H,
    intermediate_size=17408,
    rms_norm_eps=1e-6,
    full_attention_interval=4,
    linear_num_key_heads=16,
    linear_num_value_heads=48,
    linear_key_head_dim=128,
    linear_value_head_dim=128,
    linear_conv_kernel_dim=4,
    num_key_value_heads=4,
    num_attention_heads=24,
    head_dim=256,
    attention_bias=False,
    max_position_embeddings=262144,
    rope_parameters={"mrope_interleaved": True, "mrope_section": [11, 11, 10],
                     "partial_rotary_factor": 0.25, "rope_theta": 1e7,
                     "rope_type": "default"},
)

block = Qwen3_5DecoderLayer(args, 17)

# ---- linear_attn weights (GDN) ----
la = block.linear_attn
KV = 16 * 128       # 2048
VH = 48 * 128       # 6144
conv_dim = 2 * KV + VH  # 10240
la.in_proj_qkv = ql(LAYER + "linear_attn.in_proj_qkv", 2 * KV + VH, H)
la.in_proj_z = ql(LAYER + "linear_attn.in_proj_z", VH, H)
la.in_proj_b = ql(LAYER + "linear_attn.in_proj_b", 48, H)
la.in_proj_a = ql(LAYER + "linear_attn.in_proj_a", 48, H)
la.out_proj = ql(LAYER + "linear_attn.out_proj", H, VH)

# conv1d: checkpoint weight [conv_dim, kernel]; MLX expects (out, k, in//groups)
cw = load(LAYER + "linear_attn.conv1d.weight").reshape(conv_dim, 4).astype(np.float32)
la.conv1d.weight = mx.array(cw)[:, :, None].astype(mx.bfloat16)
convd = load(LAYER + "linear_attn.A_log").reshape(48).astype(np.float32)
dt = load(LAYER + "linear_attn.dt_bias").reshape(48).astype(np.float32)
la.A_log = mx.array(convd).astype(mx.bfloat16)
la.dt_bias = mx.array(dt).astype(mx.bfloat16)
lnorm = load(LAYER + "linear_attn.norm.weight").reshape(128)
la.norm.weight = mx.array(lnorm).astype(mx.bfloat16)

# ---- mlp + norms ----
block.input_layernorm.weight = mx.array(load(LAYER + "input_layernorm.weight")).astype(mx.bfloat16)
block.post_attention_layernorm.weight = mx.array(load(LAYER + "post_attention_layernorm.weight")).astype(mx.bfloat16)
mp = block.mlp
mp.gate_proj = ql(LAYER + "mlp.gate_proj", 17408, H)
mp.up_proj = ql(LAYER + "mlp.up_proj", 17408, H)
mp.down_proj = ql(LAYER + "mlp.down_proj", H, 17408)
block.eval()

# deterministic hidden, same as tests/fwd_decode.rs
hvals = np.array([(i % 97) * 0.001 for i in range(M * H)], dtype=np.float32)
h = mx.array(hvals, mx.float32).reshape(1, M, H).astype(mx.bfloat16)
pos = mx.arange(M, dtype=mx.int32)[None, :]

out = block(h, mask=None, cache=None, position_ids=pos)
res = np.asarray(out.astype(mx.float32), dtype=np.float32).reshape(M, H)
res.astype(np.float32).tofile("/tmp/mlx_fwd_truth.raw")
np.save("/tmp/mlx_fwd_truth.npy", res)

print(f"truth block out[0,:4] = {res[0,0]:.6f} {res[0,1]:.6f} {res[0,2]:.6f} {res[0,3]:.6f}")
print("saved /tmp/mlx_fwd_truth.raw", res.shape)
