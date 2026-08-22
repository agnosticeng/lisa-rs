#!/usr/bin/env python3
"""Reference Gated Delta Net layer (layer 17) computed on real
Qwen3.8-27B-4bit weights, fed the same deterministic h as tests/gdn_decode.rs.
Faithful port of saragossa's `linear_attn` math (conv+norm+gates fused kernel,
gated-delta recurrence, rms-gate).

  qkv = in_proj_qkv(h)    z = in_proj_z(h)   a = in_proj_a(h)  b = in_proj_b(h)
  conv = silu(causal K4 conv(qkv))  ; split q|c/k|v at 2048/4096
  q_norm = rms_norm(conv_q) * 1/128 ; k_norm = rms_norm(conv_k) * 1/sqrt(128)
  beta = sigmoid(b) ; decay = exp(-exp(A_log)*softplus(a+dt_bias))
  state[vh,128,128]=0
  for t in 0..M: state*=decay[t,vh]; sk=state@k_norm[t,kh]; delta=(v-sk)*beta[t,vh]
                 state+=k_norm[:,None]*delta; y[t]=state@q_norm[t,kh]  (kh=vh//3)
  gated = rms_norm(norm.weight)(y) * silu(z)
  out = out_proj(gated)

Writes /tmp/mlx_gdn_out.raw (f32 [16,5120]).
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

EPS = 1e-6
M, H = 16, 5120
KH, VH, HD = 16, 48, 128
KEYD, VALUED = KH * HD, VH * HD
CONVD = 2 * KEYD + VALUED
prefix = "language_model.model.layers.17.linear_attn."


def read_tensor(path, name):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n).decode())
        data_off = 8 + n
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        info = header[name]
        s, e = info["data_offsets"]
        return mm[data_off + s:data_off + e], info["dtype"]


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
        with open(p, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n).decode())
            if tname in header:
                return p
    raise SystemExit(f"{tname} not found")


shard = find_shard(prefix + "in_proj_qkv.weight")


def ld(name, rows, cols):
    return mx.array(load(shard, prefix + name).reshape(rows, cols))


def ld_proj(name, out, inn):
    return (ld(name + ".weight", out, inn // 8),  # affine q4 (packed cols)
            ld(name + ".scales", out, inn // 64),
            ld(name + ".biases", out, inn // 64))


qw, qs, qb = ld_proj("in_proj_qkv", CONVD, H)
z_w, z_s, z_b = ld_proj("in_proj_z", VALUED, H)
a_w, a_s, a_b = ld_proj("in_proj_a", VH, H)
b_w, b_s, b_b = ld_proj("in_proj_b", VH, H)
o_w, o_s, o_b = ld_proj("out_proj", H, VALUED)

conv_w = np.asarray(load(shard, prefix + "conv1d.weight").reshape(CONVD, 4), dtype=np.float32)
a_log = np.asarray(load(shard, prefix + "A_log").reshape(VH), dtype=np.float32)
dt_bias = np.asarray(load(shard, prefix + "dt_bias").reshape(VH), dtype=np.float32)
norm_w = np.asarray(load(shard, prefix + "norm.weight").reshape(HD), dtype=np.float32)

hvals = np.array([(i % 97) * 0.001 for i in range(M * H)], dtype=np.float32)
h = mx.array(hvals, mx.float32).reshape(M, H).astype(mx.bfloat16)


def qmm(x, w, s, b, n):
    return mx.quantized_matmul(x, w, s, b, group_size=64, bits=4)


qkv = np.asarray(qmm(h, qw, qs, qb, CONVD), dtype=np.float32)   # [M,10240]
z_in = np.asarray(qmm(h, z_w, z_s, z_b, VALUED), dtype=np.float32)
a_in = np.asarray(qmm(h, a_w, a_s, a_b, VH), dtype=np.float32)
b_in = np.asarray(qmm(h, b_w, b_s, b_b, VH), dtype=np.float32)

# causal conv (kernel 4) + silu over fused qkv
ext = np.concatenate([np.zeros((3, CONVD), np.float32), qkv], axis=0)  # [M+3, C]
conv = np.zeros((M, CONVD), np.float32)
for c in range(CONVD):
    for k in range(4):
        conv[:, c] += ext[k:k + M, c] * conv_w[c, k]
conv = conv / (1.0 + np.exp(-conv))

# rms norm + scale q/k (head 128)
def rms(x):  # x [..,128]
    return x / np.sqrt((x * x).sum(-1, keepdims=True) / 128.0 + EPS)

inv = 128.0 ** -0.5
qn_ = (rms(conv[:, :KEYD].reshape(M, KH, HD)) * (inv * inv))  # [M,KH,HD]
kn_ = (rms(conv[:, KEYD:2 * KEYD].reshape(M, KH, HD)) * inv)
v = conv[:, 2 * KEYD:].reshape(M, VH, HD)

beta = 1.0 / (1.0 + np.exp(-b_in))                                # [M,48]
dtarg = a_in + dt_bias[None, :]
sp = np.where(dtarg > 20, dtarg, np.log1p(np.exp(dtarg)))
decay = np.exp(-np.exp(a_log[None, :]) * sp)                      # [M,48]

state = np.zeros((VH, HD, HD), np.float32)
y = np.zeros((M, VH, HD), np.float32)
for t in range(M):
    for vh_ in range(VH):
        kh = vh_ // (VH // KH)
        state[vh_] = state[vh_] * decay[t, vh_]
        sk = (state[vh_] * kn_[t, kh, :][None, :]).sum(-1)
        delta = (v[t, vh_] - sk) * beta[t, vh_]
        state[vh_] = state[vh_] + kn_[t, kh, :][None, :] * delta[:, None]
        y[t, vh_] = (state[vh_] * qn_[t, kh, :][None, :]).sum(-1)

rms_y = 1.0 / np.sqrt((y * y).sum(-1, keepdims=True) / 128.0 + EPS)
gated = (y * rms_y * norm_w[None, None, :]) * (z_in.reshape(M, VH, HD) / (1.0 + np.exp(-z_in.reshape(M, VH, HD))))
gated = gated.reshape(M, VALUED).astype(np.float32)
gated_bf16 = mx.array(gated, mx.float32).astype(mx.bfloat16)

out = np.asarray(qmm(gated_bf16, o_w, o_s, o_b, H), dtype=np.float32)
np.save("/tmp/mlx_gdn_out.npy", out)
out.astype("<f4").tofile("/tmp/mlx_gdn_out.raw")
print(f"gdn ref out[0,:4] = {out[0,0]:.6f} {out[0,1]:.6f} {out[0,2]:.6f} {out[0,3]:.6f}")
print("saved /tmp/mlx_gdn_out.raw", out.shape)