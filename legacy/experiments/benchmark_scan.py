"""Work 1 baseline: benchmark full HMN training step speed + SSM scan speed + reconstruction.
Runs before the chunked scan change to establish reference numbers.
Configs from doc 16.1: dim64/4L/bs6, dim96/6L/bs8, dim128/8L/bs8.
Also measures SelectiveSSM pure-scan time (the target of the optimization).
"""
import sys, time, torch
sys.path.insert(0, '/home/yonoob/projects/ReTop')
import torch.nn as nn
from hmn_v2 import HMN, HelixCouplingBlock, ReversibleFunction

torch.set_num_threads(2)
torch.manual_seed(0)

VOCAB = 120
CONFIGS = [(64, 4, 6), (96, 6, 8), (128, 8, 8)]
N_STEPS = 12
T_SEQ = 64


def model_step_time(dim, layers, bs):
    m = HMN(VOCAB, dim, state_dim=dim // 8, n_layers=layers, n_experts=16, top_k=2,
            n_mem_cells=32, mem_top_k=4)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    x = torch.randint(0, VOCAB, (bs, T_SEQ))
    # warmup
    for _ in range(3):
        out = m(x); loss = out.mean(); loss.backward()
        opt.step(); opt.zero_grad()
    start = time.perf_counter()
    for _ in range(N_STEPS):
        out = m(x)
        loss = out.mean()
        loss.backward()
        opt.step()
        opt.zero_grad()
    dt = (time.perf_counter() - start) / N_STEPS
    return dt


def ssm_scan_time(dim, layers, bs):
    """Time ONLY the SSM scans inside the coupling blocks (the sequential loop)."""
    blocks = [HelixCouplingBlock(dim, dim // 8) for _ in range(layers)]
    h = torch.randn(bs, T_SEQ, dim)
    for b in blocks:  # warmup
        y = b.forward(h); y.sum().backward()
    start = time.perf_counter()
    for _ in range(N_STEPS):
        y = h
        for b in blocks:
            y = b.forward(y)
        y.sum().backward()
    dt = (time.perf_counter() - start) / N_STEPS
    return dt


def reconstruction_error(dim, layers, bs):
    blocks = [HelixCouplingBlock(dim, dim // 8) for _ in range(layers)]
    h = torch.randn(bs, T_SEQ, dim)
    hf = h
    with torch.no_grad():
        for b in blocks:
            hf = b.forward(hf)
        hr = hf
        for b in reversed(blocks):
            hr = b.inverse(hr)
        err = (hr - h).abs().max().item()
    return err


if __name__ == '__main__':
    print(f'torch threads={torch.get_num_threads()}, seq_len={T_SEQ}')
    for dim, layers, bs in CONFIGS:
        t_full = model_step_time(dim, layers, bs)
        t_scan = ssm_scan_time(dim, layers, bs)
        rec = reconstruction_error(dim, layers, bs)
        print(f'dim{dim}/L{layers}/bs{bs}: full_step={t_full*1000:.1f}ms '
              f'scan_only={t_scan*1000:.1f}ms recon_err={rec:.2e}')
