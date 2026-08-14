"""retop_gui — Gradio web UI for the ReTop (HMN v3) slot-copy pipeline.

English UI, four tabs: DATA / TRAIN / CHAT / VERIFY.

Design: the GUI is a thin wrapper. Generating data and training spawn the exact
CLI scripts (gen_slots.py, retop.py train) as subprocesses and stream their
stdout into the UI — the training recipe stays a single source of truth and the
GUI cannot drift from it. Chat and Verify run in-process (read-only) on the same
functions the CLI uses (hmn/recipe.decode_v33, slot_v33_seed42.run_guards).

Usage:
    pip install gradio
    python retop_gui.py                 # opens http://localhost:7860
    retop-gui                           # same (installed console script)
"""
import json
import os
import re
import subprocess
import sys
import threading

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import gradio as gr
import torch

from tokenizers import Tokenizer

from hmn.recipe import decode_v33, make_chat_ids
from retop import SPECS, build_model as retop_build_model

DEFAULT_TOK = os.path.join(ROOT, "retop_tokenizer.json")
DEFAULT_CKPT = os.path.join(ROOT, "hmn_v33.pt")

_METRIC = re.compile(
    r"^step\s+(\d+)\s+loss=([\d.]+)\s+seen=([\d.]+)\(g[\d.]+\)"
    r"\s+unseen_blend=([\d.]+)\(g[\d.]+,gen[\d.]+\)(?:\s+hard=([\d.]+))?")

# --- shared training state (subprocess + reader thread) -----------------------

STATE = {"proc": None, "lines": [], "metrics": [], "running": False,
         "exit": None}
_LOCK = threading.Lock()


def _spawn(cmd):
    """Run cmd in a subprocess; stream stdout lines + parsed metrics into STATE."""
    with _LOCK:
        STATE["lines"] = []
        STATE["metrics"] = []
        STATE["exit"] = None
        STATE["running"] = True
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, cwd=ROOT)
        STATE["proc"] = proc

    def reader():
        for line in proc.stdout:
            line = line.rstrip("\n")
            with _LOCK:
                STATE["lines"].append(line)
                if len(STATE["lines"]) > 4000:
                    STATE["lines"] = STATE["lines"][-4000:]
                m = _METRIC.match(line)
                if m:
                    STATE["metrics"].append({"step": int(m.group(1)),
                                             "loss": float(m.group(2)),
                                             "seen": float(m.group(3)),
                                             "unseen": float(m.group(4)),
                                             "hard": float(m.group(5)) if m.group(5) else None})
        rc = proc.wait()
        with _LOCK:
            STATE["exit"] = rc
            STATE["running"] = False

    threading.Thread(target=reader, daemon=True).start()


def _fmt(v):
    return f"{v:.3f}" if isinstance(v, (int, float)) else "—"


def _svg_chart(points, width=760, height=150):
    """Minimal dependency-free SVG line chart for seen/unseen/loss over steps."""
    if not points or len(points) < 2:
        return "<div style='color:#999;font-family:monospace'>no eval points yet</div>"
    margin = 28
    w, h = width, height
    pw, ph = w - 2 * margin, h - 2 * margin
    mx = max(p["step"] for p in points)
    y_min, y_max = 0.0, 1.0
    los, his = (v for series in ("loss",) for v in ([p["loss"] for p in points if p["loss"] is not None] or [0]))
    if points:
        los = min((p["loss"] for p in points if p["loss"] is not None), default=0.0)
        his = max((p["loss"] for p in points if p["loss"] is not None), default=1.0)
        y_min, y_max = min(0.0, los * 0.9), max(1.0, his * 1.1)
    span = max(y_max - y_min, 1e-12)

    def xy(step, val):
        return (margin + pw * step / max(mx, 1),
                margin + ph * (1 - (val - y_min) / span))

    colors = {"seen": "#2e9e5b", "unseen": "#1f77b4", "loss": "#c0392b"}
    parts = [f"<line x1='{margin}' y1='{margin+ph}' x2='{w-margin}' y2='{margin+ph}' stroke='#ccc'/>",
             f"<line x1='{margin}' y1='{margin+ph/2}' x2='{w-margin}' y2='{margin+ph/2}' stroke='#eee'/>"]
    for name, color in colors.items():
        pts = [(p["step"], p[name]) for p in points if p.get(name) is not None]
        if len(pts) >= 2:
            poly = " ".join(f"{xy(a, b)[0]:.0f},{xy(a, b)[1]:.0f}" for a, b in pts)
            parts.append(f"<polyline points='{poly}' fill='none' stroke='{color}' stroke-width='2'/>")
            last = xy(*pts[-1])
            parts.append(f"<text x='{w-margin-2}' y='{last[1]}' fill='{color}' "
                         f"font-size='11' text-anchor='end'>{name}</text>")
    return f"<svg width='{w}' height='{h}'>{''.join(parts)}</svg>"


def _state_html():
    if STATE["running"]:
        return "<b>… training …</b>"
    if STATE.get("exit") is not None:
        return f"<b>exit code: {STATE['exit']}</b>"
    return "<b>idle</b>"


# --- DATA tab ----------------------------------------------------------------

def generate_data(template, kind, n_seen, n_unseen, seed, out_path):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    cmd = [sys.executable, "gen_slots.py", "--out", out_path, "--n-seen", str(int(n_seen)),
           "--n-unseen", str(int(n_unseen)), "--kind", kind, "--template", template,
           "--seed", str(int(seed))]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        return "", f"<pre style='color:#c0392b'>{r.stdout}{r.stderr}</pre>", ""
    val_path = out_path[:-6] + "-val.jsonl" if out_path.endswith(".jsonl") else out_path + "-val.jsonl"

    def rows(p, n=3):
        out = []
        with open(p, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= n:
                    break
                m = json.loads(line)["messages"]
                out.append((m[0]["content"], m[1]["content"]))
        return out

    md = ["**Train (first rows):**", "| prompt | answer |", "|---|---|"]
    for a, b in rows(out_path):
        md.append(f"| `{a}` | `{b}` |")
    md.append("")
    md.append(f"**Val / unseen (first rows, no overlap with train):**")
    for a, b in rows(val_path):
        md.append(f"- `{a}`")
    log = f"generated: `{out_path}` (+ {val_path}). {r.stdout.strip()}"
    return "\n".join(md), log.replace("\n", "<br>"), out_path


def upload_data(file):
    if not file:
        return "", "", ""
    p = file.name if hasattr(file, "name") else str(file)
    val = p[:-6] + "-val.jsonl" if p.endswith(".jsonl") else p + "-val.jsonl"
    md = f"Using uploaded data: `{p}`\n\nval companion exists: `{os.path.exists(val)}`"
    return md, p, p


# --- TRAIN tab ---------------------------------------------------------------

def start_training(data_path, tok_path, spec, steps, lr, seed, out_path):
    if STATE["running"]:
        return "training already running", _state_html(), "—", "—", "—", "—", \
            _svg_chart([]), gr.update(interactive=True), gr.update(interactive=False)
    if not os.path.exists(data_path):
        return f"data file not found: {data_path}", _state_html(), "—", "—", "—", "—", \
            _svg_chart([]), gr.update(interactive=True), gr.update(interactive=False)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    cmd = [sys.executable, "retop.py", "train",
           "--data", data_path, "--tok", tok_path, "--out", out_path,
           "--arch", "v3", "--spec", spec, "--steps", str(int(steps)),
           "--lr", str(float(lr)), "--seed", str(int(seed))]
    _spawn(cmd)
    return "training started", "<b>… training …</b>", "—", "—", "—", "—", \
        _svg_chart([]), gr.update(interactive=False), gr.update(interactive=True)


def stop_training():
    p = STATE.get("proc")
    if p and p.poll() is None:
        p.terminate()
        try:
            p.wait(timeout=8)
        except subprocess.TimeoutExpired:
            p.kill()
    return _state_html(), gr.update(interactive=True), gr.update(interactive=False)


def poll_training():
    with _LOCK:
        log = "\n".join(STATE["lines"])
        metrics = list(STATE["metrics"])
    m = metrics[-1] if metrics else {}
    return (_state_html(), _fmt(m.get("loss")), _fmt(m.get("seen")),
            _fmt(m.get("unseen")), _fmt(m.get("hard")), _svg_chart(metrics),
            gr.update(interactive=not STATE["running"]),
            gr.update(interactive=STATE["running"]))


# --- CHAT tab ----------------------------------------------------------------

CHAT = {"ckpt": None, "model": None, "tok": None}


def load_chat(ckpt_path, tok_path, dim, layers, gate_bias, max_new, mode):
    if not os.path.exists(ckpt_path):
        return f"checkpoint not found: {ckpt_path}", ""
    if not os.path.exists(tok_path):
        return f"tokenizer not found: {tok_path}", ""
    tok = Tokenizer.from_file(tok_path)
    cfg = {"vocab": tok.get_vocab_size(), "arch": "v3", "seed": 0, "spec": "",
           "tokenizer": tok_path, "dim": int(dim), "layers": int(layers),
           "moe": False, "gate_bias": float(gate_bias)}
    side = ckpt_path + ".json"
    if os.path.exists(side):
        try:
            with open(side) as f:
                cfg.update({k: v for k, v in json.load(f).items()
                            if k in ("arch", "dim", "layers", "moe", "gate_bias",
                                     "spec", "tokenizer", "vocab")})
        except (OSError, ValueError):
            pass
        if cfg.get("spec") in SPECS:
            cfg.update({k: v for k, v in SPECS[cfg["spec"]].items()
                        if k in ("dim", "layers", "moe", "gate_bias")})
    try:
        model = retop_build_model(cfg.get("arch", "v3"), cfg, tok)
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        model.eval()
    except Exception as e:
        return f"failed to load: {e}", ""
    CHAT.update(ckpt=ckpt_path, model=model, tok=tok)
    n = sum(p.numel() for p in model.parameters())
    info = (f"✅ loaded v3 ({n:,} params, dim={cfg['dim']}, L={cfg['layers']}, "
            f"gate_bias={cfg['gate_bias']}) from `{ckpt_path}`")
    return info, info


def chat_generate(prompt, max_new, mode):
    if CHAT["model"] is None:
        return "load a model first (CHAT tab → Load model)", ""
    if not prompt.strip():
        return "enter a prompt", ""
    try:
        ids = make_chat_ids(CHAT["tok"], prompt.strip())
        text, gate, ngen = decode_v33(CHAT["model"], CHAT["tok"], ids,
                                      max_new=int(max_new), mode=mode)
    except Exception as e:
        return f"error: {e}", ""
    wraps = (f"\n\n> gate avg **{gate:.2f}** | generate tokens **{int(ngen)}** — "
             f"model trained only on the slot template; general chat may look random")
    return text.strip() or "(eos immediately)", text.strip() + wraps


# --- VERIFY tab --------------------------------------------------------------

def verify_guards(ckpt_path):
    sys.path.append(os.path.join(ROOT, "experiments", "verified"))
    try:
        import slot_v33_seed42 as v
    except Exception as e:
        return f"guardrail import failed: {e}", None
    if not os.path.exists(ckpt_path):
        return f"checkpoint not found: {ckpt_path}", None
    try:
        r = v.run_guards(ckpt_path)
    except Exception as e:
        return (f"guardrail failed (needs a v3 slot-copy checkpoint, D96/L3): "
                f"{e}"), None
    rows = [["unseen blend", f"{r['blend'][0]:.3f}",
             "PASS" if r["blend"][0] >= 1.0 else "FAIL"],
            ["unseen hard", f"{r['hard']:.3f}",
             "PASS" if r["hard"] >= 1.0 else "FAIL"],
            [f"structural {r['structural'][2]!r}", f"{r['structural'][0]:.3f}",
             "PASS" if r["structural"][0] >= 1.0 else "FAIL"]]
    for row in r["matrix"]:
        rows.append([f"{row['name']} ({row['ok_n']}/{row['total']})",
                     f"{row['acc']:.3f}", "probe"])
    status = ("**ALL GUARDS PASSED** ✅" if r["ok"] else "**GUARDRAIL FAILED** ❌")
    return status, rows


def run_v4_guardrail():
    """v4 matrix (M7): re-runs the consolidated v4 guardrail script. Blocks the
    UI for ~2-3 min (subprocess, captured output returned as markdown)."""
    script = os.path.join(ROOT, "experiments", "verified", "v4_guardrail.py")
    if not os.path.exists(script):
        return "v4_guardrail.py not found"
    import subprocess as _sp
    cp = _sp.run([sys.executable, script], cwd=ROOT, capture_output=True, text=True)
    tail = (cp.stdout + cp.stderr).strip().splitlines()
    brief = "\n".join(line for line in tail
                      if "PASSED" in line or "OK" in line or "FAILED" in line)
    return f"**returncode {cp.returncode}**\n```\n{brief}\n```"


# --- layout ------------------------------------------------------------------

def build():
    with gr.Blocks(title="ReTop — HMN v3.3 slot-copy", theme=gr.themes.Default()) as demo:
        gr.Markdown("# ReTop — Helix Memory Network (HMN v3.3)\n"
                    "CPU-trainable slot-copy. **Data → Train → Chat → Verify.** "
                    "See `docs/hmn_v3_design.md`.")

        with gr.Tab("DATA"):
            with gr.Row():
                with gr.Column(scale=1):
                    template = gr.Textbox("pip install {slot}", label="Template",
                                          info="{slot} is replaced per value")
                    kind = gr.Dropdown(["pkg3", "pkg4", "pkg5", "alnum", "repeat"],
                                       value="pkg3", label="Slot kind")
                    with gr.Row():
                        n_seen = gr.Number(600, label="Seen (train)", precision=0)
                        n_unseen = gr.Number(400, label="Unseen (val)", precision=0)
                    seed = gr.Number(0, label="Seed", precision=0)
                    out_path = gr.Textbox(os.path.join(ROOT, "data", "slots.jsonl"),
                                          label="Output .jsonl path")
                    gen_btn = gr.Button("✨ Generate data", variant="primary")
                    data_path_in = gr.Textbox("", label="Data file (train .jsonl)")
                    gen_md = gr.Markdown("click Generate…")
                    gen_log = gr.HTML("")
                with gr.Column(scale=1):
                    upload = gr.File(label="or upload your own .jsonl chat pairs",
                                     file_types=[".jsonl"])
                    upload_md = gr.Markdown("")
        with gr.Tab("TRAIN"):
            with gr.Row():
                with gr.Column(scale=1):
                    tok_path = gr.Textbox(DEFAULT_TOK, label="Tokenizer .json")
                    with gr.Row():
                        spec = gr.Dropdown(["auto"] + list(SPECS), value="small", label="Spec")
                        steps = gr.Number(1400, label="Steps", precision=0)
                    with gr.Row():
                        lr = gr.Number(3e-4, label="Learning rate")
                        seed_tr = gr.Number(0, label="Seed", precision=0)
                    out_in = gr.Textbox(os.path.join(ROOT, "model.pt"), label="Checkpoint out")
                    with gr.Row():
                        start_btn = gr.Button("▶ Start training", variant="primary")
                        stop_btn = gr.Button("■ Stop", interactive=False)
                with gr.Column(scale=1):
                    status_html = gr.HTML("<b>idle</b>")
                    with gr.Row():
                        loss_out = gr.Textbox("—", label="loss", interactive=False)
                        seen_out = gr.Textbox("—", label="seen", interactive=False)
                        unseen_out = gr.Textbox("—", label="unseen ★", interactive=False)
                        hard_out = gr.Textbox("—", label="hard", interactive=False)
                    chart_html = gr.HTML(_svg_chart([]))
            log_out = gr.Textbox("", label="training log", lines=16, max_lines=16,
                                 interactive=False)
        with gr.Tab("CHAT"):
            with gr.Row():
                with gr.Column(scale=1):
                    ckpt_in = gr.Textbox(DEFAULT_CKPT, label="Checkpoint (.pt)")
                    tok_chat = gr.Textbox(DEFAULT_TOK, label="Tokenizer .json")
                    with gr.Row():
                        dim_c = gr.Number(96, label="dim", precision=0)
                        layers_c = gr.Number(3, label="layers", precision=0)
                        gb_c = gr.Number(-1.0, label="gate_bias")
                    with gr.Row():
                        max_new = gr.Number(16, label="max new tokens", precision=0)
                        mode_c = gr.Dropdown(["blend", "hard", "copy"], value="blend",
                                             label="decode mode")
                    load_btn = gr.Button("Load model", variant="primary")
                with gr.Column(scale=1):
                    load_md = gr.Markdown("load a checkpoint…")
            prompt = gr.Textbox("pip install pkg042", label="Prompt", lines=2)
            chat_btn = gr.Button("▶ Generate", variant="primary")
            answer = gr.Markdown("")
        with gr.Tab("VERIFY"):
            ckpt_v = gr.Textbox(DEFAULT_CKPT, label="Checkpoint (.pt)")
            ver_btn = gr.Button("🔍 Run guardrail", variant="primary")
            ver_md = gr.Markdown("")
            ver_tbl = gr.Dataframe(headers=["check", "acc", "status"],
                                   datatype=["str", "str", "str"], interactive=False)
            gr.Markdown("**v4 matrix** — one command re-runs every v4 gate "
                        "(test_hmn, M1 parity, seed-42 40/40, M8 smoke). "
                        "Takes ~2-3 min CPU.")
            v4_btn = gr.Button("🚦 Run full v4 guardrail", variant="secondary")
            v4_md = gr.Markdown("")
        # --- event wiring ---
        gen_btn.click(generate_data, [template, kind, n_seen, n_unseen, seed, out_path],
                      [gen_md, gen_log, data_path_in], queue=True)
        upload.change(upload_data, upload, [upload_md, gen_log, data_path_in], queue=True)
        start_btn.click(start_training, [data_path_in, tok_path, spec, steps, lr, seed_tr, out_in],
                        [status_html, status_html, loss_out, seen_out, unseen_out, hard_out,
                         chart_html, start_btn, stop_btn], queue=True)
        stop_btn.click(stop_training, None, [status_html, start_btn, stop_btn], queue=True)
        gr.Timer(1.0).tick(poll_training, None,
                           [status_html, loss_out, seen_out, unseen_out, hard_out,
                            chart_html, start_btn, stop_btn], show_progress=False)
        load_btn.click(load_chat, [ckpt_in, tok_chat, dim_c, layers_c, gb_c, max_new, mode_c],
                       [load_md, load_md], queue=True)
        chat_btn.click(chat_generate, [prompt, max_new, mode_c], [answer, answer], queue=True)
        ver_btn.click(verify_guards, ckpt_v, [ver_md, ver_tbl], queue=True)
        v4_btn.click(run_v4_guardrail, None, v4_md, queue=True)
    return demo


def main():
    demo = build()
    demo.queue()
    demo.launch(server_name="127.0.0.1")


if __name__ == "__main__":
    main()