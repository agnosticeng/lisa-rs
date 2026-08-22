#!/usr/bin/env python3
"""Reference MLP block (layer 17) computed with MLX on real Qwen3.8-27B-4bit
weights, fed the same deterministic h as tests/mlp_decode.rs. Dumps the output
row so the Rust engine can anchor correctness.

Prints unit tests style "mlp ref out[0,:4] = ..." and writes /tmp/mlx_out.npy
(f32, [16,5120]).
"""
import os, struct, json, glob, mmap
import numpy as np
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import mlx_env as _mlx_env
_mlx_env.reexec()
import mlx.core as mx

HOME = os.path.expanduser("~")
CACHE = f"{HOME}/.cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-4bit/snapshots"
shards = sorted(glob.glob(f"{CACHE}/*/*.safetensors"))


def read_tensor(path, name):
    # parse safetensors header, mmap the file, return raw bytes + dtype
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n).decode())
        data_off = 8 + n
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        info = header[name]
        start, end = info["data_offsets"]
        return mm[data_off + start : data_off + end], info["dtype"]


def load(path, name):
    raw, dt = read_tensor(path, name)
    if dt == "U32":
        return np.frombuffer(raw, dtype="<u4").copy()
    if dt == "BF16":
        b = np.frombuffer(raw, dtype="<u2").copy()
        return (b.astype(np.uint32) << 16).view(np.float32).copy()
    raise ValueError(dt)


def find_shard(tname):
    for p in shards:
        try:
            with open(p, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(n).decode())
                if tname in header:
                    return p
        except Exception:
            continue
    raise SystemExit(f"{tname} not found")


M, H, INTER = 16, 5120, 17408
prefix = "language_model.model.layers.17.mlp."
shard = find_shard(prefix + "gate_proj.weight")


def ld(name, rows, cols):
    a = load(shard, prefix + name)
    return mx.array(a.reshape(rows, cols))


gw = ld("gate_proj.weight", INTER, H // 8)
gs = ld("gate_proj.scales", INTER, H // 64)
gb = ld("gate_proj.biases", INTER, H // 64)
uw = ld("up_proj.weight", INTER, H // 8)
us = ld("up_proj.scales", INTER, H // 64)
ub = ld("up_proj.biases", INTER, H // 64)
dw = ld("down_proj.weight", H, INTER // 8)
ds = ld("down_proj.scales", H, INTER // 64)
db = ld("down_proj.biases", H, INTER // 64)

# same deterministic h as tests/mlp_decode.rs
hvals = np.array([(i % 97) * 0.001 for i in range(M * H)], dtype=np.float32)
h = mx.array(hvals, mx.float32).reshape(M, H).astype(mx.bfloat16)

gate = mx.quantized_matmul(h, gw, gs, gb, group_size=64, bits=4)
up = mx.quantized_matmul(h, uw, us, ub, group_size=64, bits=4)
si = (mx.sigmoid(gate) * gate * up).astype(mx.bfloat16)
out = mx.quantized_matmul(si, dw, ds, db, group_size=64, bits=4)

o = np.asarray(out, dtype=np.float32)
np.save("/tmp/mlx_out.npy", o)
o.astype("<f4").tofile("/tmp/mlx_out.raw")
print(f"mlp ref out[0,:4] = {o[0,0]:.8f} {o[0,1]:.8f} {o[0,2]:.8f} {o[0,3]:.8f}")
print("saved /tmp/mlx_out.raw", o.shape)