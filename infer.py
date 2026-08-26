"""Inference for HMN checkpoints: load a trained model + tokenizer and generate.

Works for all architectures (v2 HMN, HMN_Option1, v3 HMN3) from the same CLI.
Checkpoint must come from a training script that saves `model.state_dict()` and
must be invoked with the SAME architecture hyperparameters used at train time.

Usage:
  # one-shot
  python infer.py --checkpoint ckpt.pt --prompt "pip install numpy"
  # interactive REPL
  python infer.py --checkpoint ckpt.pt --interactive --max-new 64
  # different architecture / config
  python infer.py --checkpoint v3.pt --arch v3 --dim 96 --layers 3 --gate-bias -2.0
"""
import argparse
import os

import torch

from tokenizers import Tokenizer

from hmn import HMN3, HMN3AttentionWR
from hmn.checkpoint import load_compat
from hmn.recipe import decode_v33, make_chat_ids, resolve_device

ROOT = os.path.dirname(os.path.abspath(__file__))
EOS = "</s>"
ASI = "<|assistant|>"


def build_tokenizer(path):
    return Tokenizer.from_file(path)


def build_model(arch, args, tok):
    vocab = tok.get_vocab_size()
    if arch == "v3":
        return HMN3(vocab, dim=args.dim, state_dim=args.state, n_layers=args.layers,
                    use_moe=args.moe, gate_bias=args.gate_bias,
                    use_think=args.think, k_max=args.k_max,
                    asi_id=tok.token_to_id(ASI),
                    exact_blend=args.exact_blend)
    if arch == "attention":
        return HMN3AttentionWR(vocab, dim=args.dim, n_layers=args.layers,
                               use_moe=args.moe, gate_bias=args.gate_bias,
                               asi_id=tok.token_to_id(ASI))
    raise ValueError(f"unknown --arch {arch!r}")


def generate(model, tok, prompt_ids, max_new, arch="v2", device=None):
    """Greedy decode. v3 uses hmn/recipe.decode_v33 (blend of gen+copy with the
    boundary rule); legacy v2/option1 use softmax argmax."""
    dev = resolve_device(device)
    return decode_v33(model, tok, prompt_ids, max_new, device=dev)[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help=".pt state_dict to load")
    ap.add_argument("--arch", default="v3", choices=["v3", "attention"])
    ap.add_argument("--tok", default=os.path.join(ROOT, "retop_tokenizer.json"))
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--state", type=int, default=8)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--mem-cells", type=int, default=8)
    ap.add_argument("--mem-interval", type=int, default=2)
    ap.add_argument("--combined", action="store_true")
    ap.add_argument("--exempt-combined", action="store_true")
    ap.add_argument("--n-pairs", type=int, default=None)
    ap.add_argument("--moe", action="store_true")
    ap.add_argument("--gate-bias", type=float, default=0.0)
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--k-max", type=int, default=4)
    ap.add_argument("--exact-blend", action="store_true",
                    help="v6 M1-B: legacy dense blended-logits oracle instead of "
                         "the IRStats index path")
    ap.add_argument("--prompt", default=None, help="one-shot prompt")
    ap.add_argument("--interactive", action="store_true", help="REPL loop")
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None,
                   help="compute device (auto-detect default; see resolve_device)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = resolve_device(args.device)
    tok = build_tokenizer(args.tok)
    model = build_model(args.arch, args, tok)
    try:
        load_compat(model, args.checkpoint, device=dev)
    except RuntimeError as e:
        raise SystemExit(
            f"checkpoint does not match --arch {args.arch} config "
            f"(dim={args.dim}, layers={args.layers}, ...). "
            f"Retrain/load with the same hyperparameters used at train time.\n{e}") from e
    model.to(dev).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"loaded {args.arch} ({n_params:,} params) from {args.checkpoint}", flush=True)

    if args.interactive:
        while True:
            try:
                user = input(">>> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not user:
                continue
            print(generate(model, tok, make_chat_ids(tok, user), args.max_new,
                          args.arch, device=dev))
    elif args.prompt:
        print(generate(model, tok, make_chat_ids(tok, args.prompt), args.max_new,
                       args.arch, device=dev))
    else:
        ap.error("need --prompt or --interactive")


if __name__ == "__main__":
    main()
