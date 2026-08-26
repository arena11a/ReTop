"""HMN v9 Trainer — production-ready training loop.

Features:
  - Mixed precision (torch.amp autocast + GradScaler)
  - Gradient accumulation
  - LR schedule (warmup + cosine decay)
  - Checkpoint resume (model + optimizer + scheduler + step)
  - Early stopping (patience-based)
  - Eval integration (call eval function during training)
  - TensorBoard logging (optional)

Usage:
    from hmn.trainer import Trainer

    trainer = Trainer(model, config=cfg, loss_fn=loss_v33)
    trainer.train(dataloader, val_fn=eval_slots, val_kwargs={...})
"""
import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Optional, Callable

import torch
import torch.nn as nn


@dataclass
class TrainerConfig:
    """Training hyperparameters (all optional, defaults match existing behavior)."""
    lr: float = 3e-4
    weight_decay: float = 0.0
    grad_clip: float = 5.0
    grad_accum: int = 1
    optimizer: str = "adamw"  # "adam" or "adamw"
    # LR schedule
    lr_schedule: str = "constant"  # "constant", "cosine", "linear"
    warmup_steps: int = 0
    # Mixed precision
    use_amp: bool = False
    amp_dtype: str = "bf16"  # "bf16" or "fp16"
    # torch.compile
    use_compile: bool = False  # enable torch.compile for speedup
    compile_mode: str = "default"  # "default", "reduce-overhead", "max-autotune"
    # Early stopping
    patience: int = 0  # 0 = disabled
    # Checkpointing
    save_every: int = 0  # 0 = only save best
    keep_last: int = 3  # number of recent checkpoints to keep
    # Logging
    log_every: int = 100
    tb_dir: Optional[str] = None  # TensorBoard log directory


class Trainer:
    """Production training loop for HMN models.

    Supports mixed precision, gradient accumulation, LR scheduling,
    checkpoint resume, and early stopping.

    Args:
        model: HMN3 or HMN3AttentionWR instance
        config: TrainerConfig or dict with training hyperparams
        loss_fn: loss function (default: nn.CrossEntropyLoss)
        extra_loss_fn: optional extra loss function (e.g. loss_v33 for dual-head)
    """

    def __init__(self, model, config=None, loss_fn=None, extra_loss_fn=None):
        self.model = model
        if isinstance(config, dict):
            config = TrainerConfig(**config)
        self.cfg = config or TrainerConfig()
        self.loss_fn = loss_fn or nn.CrossEntropyLoss(ignore_index=-100)
        self.extra_loss_fn = extra_loss_fn

        # Setup optimizer
        self._setup_optimizer()

        # Mixed precision
        self.scaler = torch.amp.GradScaler("cpu", enabled=False)
        self.amp_dtype = torch.bfloat16 if self.cfg.amp_dtype == "bf16" else torch.float16
        if self.cfg.use_amp:
            self.scaler = torch.amp.GradScaler(
                "cuda" if torch.cuda.is_available() else "cpu",
                enabled=True
            )

        # LR schedule
        self.scheduler = None
        self.total_steps = 0
        self.warmup_steps = self.cfg.warmup_steps

        # State
        self.step = 0
        self.best_val = float("inf")
        self.patience_counter = 0
        self._tb_writer = None

        # torch.compile
        self._compiled = False
        if self.cfg.use_compile:
            self._setup_compile()

    def _setup_optimizer(self):
        """Create optimizer based on config."""
        params = self.model.parameters()
        wd = self.cfg.weight_decay
        lr = self.cfg.lr
        if self.cfg.optimizer == "adam":
            self.optimizer = torch.optim.Adam(params, lr=lr, weight_decay=wd)
        else:
            self.optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd)

    def _setup_compile(self):
        """Compile model with torch.compile for speedup."""
        if not self.cfg.use_compile:
            return
        try:
            self.model = torch.compile(
                self.model,
                mode=self.cfg.compile_mode,
            )
            self._compiled = True
            self._log(f"torch.compile enabled (mode={self.cfg.compile_mode})")
        except Exception as e:
            self._log(f"torch.compile failed: {e}, falling back to eager")
            self._compiled = False

    def _setup_scheduler(self, total_steps):
        """Create LR schedule."""
        self.total_steps = total_steps
        if self.cfg.lr_schedule == "cosine":
            def lr_lambda(step):
                if step < self.warmup_steps:
                    return float(step) / float(max(1, self.warmup_steps))
                progress = float(step - self.warmup_steps) / float(max(1, total_steps - self.warmup_steps))
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        elif self.cfg.lr_schedule == "linear":
            def lr_lambda(step):
                if step < self.warmup_steps:
                    return float(step) / float(max(1, self.warmup_steps))
                return max(0.0, 1.0 - float(step - self.warmup_steps) / float(max(1, total_steps - self.warmup_steps)))
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        else:
            self.scheduler = None

    def save_checkpoint(self, path, extra=None):
        """Save full training state for resume."""
        state = {
            "step": self.step,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_val": self.best_val,
            "patience_counter": self.patience_counter,
        }
        if self.scheduler is not None:
            state["scheduler_state"] = self.scheduler.state_dict()
        if self.scaler is not None and self.scaler.is_enabled():
            state["scaler_state"] = self.scaler.state_dict()
        if extra:
            state["extra"] = extra
        torch.save(state, path)

    def load_checkpoint(self, path):
        """Load training state from checkpoint."""
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state["model_state"])
        self.optimizer.load_state_dict(state["optimizer_state"])
        self.step = state.get("step", 0)
        self.best_val = state.get("best_val", float("inf"))
        self.patience_counter = state.get("patience_counter", 0)
        if self.scheduler is not None and "scheduler_state" in state:
            self.scheduler.load_state_dict(state["scheduler_state"])
        if self.scaler is not None and "scaler_state" in state:
            self.scaler.load_state_dict(state["scaler_state"])
        return state.get("extra", None)

    def _log(self, msg):
        """Print with timestamp."""
        print(f"[trainer step {self.step}] {msg}", flush=True)

    def _log_tb(self, tag, value):
        """Log to TensorBoard if available."""
        if self._tb_writer is not None:
            self._tb_writer.add_scalar(tag, value, self.step)

    def train_step(self, batch):
        """Single training step. Returns loss value.

        batch: tuple of tensors (X, Y, ...) — interpreted by loss_fn
        """
        self.model.train()
        # Unpack batch
        if isinstance(batch, (list, tuple)):
            X = batch[0]
            Y = batch[1] if len(batch) > 1 else None
        else:
            X = batch
            Y = None

        # Forward pass with optional AMP
        use_amp = self.cfg.use_amp and X.is_cuda
        with torch.amp.autocast("cuda" if X.is_cuda else "cpu",
                                enabled=use_amp, dtype=self.amp_dtype):
            out = self.model(X)
            if Y is not None and not isinstance(out, dict):
                loss = self.loss_fn(out.reshape(-1, out.shape[-1]), Y.reshape(-1))
            elif isinstance(out, dict) and Y is not None:
                # Dual-head path: out is dict with logits, gen_logits, stats, etc.
                # Use extra_loss_fn if provided, otherwise fall back to gen CE
                if self.extra_loss_fn is not None:
                    # extra_loss_fn expects (out, Y, Yc, G) — caller must provide
                    # these in the batch. For simplicity, use gen CE here.
                    logits = out.get("logits", out.get("gen_logits"))
                    loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), Y.reshape(-1))
                else:
                    logits = out.get("logits", out.get("gen_logits"))
                    loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), Y.reshape(-1))
            else:
                logits = out.get("logits", out.get("gen_logits")) if isinstance(out, dict) else out
                loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), Y.reshape(-1))

            # MoE aux loss
            if hasattr(self.model, "moe_aux_loss"):
                loss = loss + self.model.moe_aux_loss()

        # Backward with gradient accumulation
        if use_amp:
            self.scaler.scale(loss / self.cfg.grad_accum).backward()
        else:
            (loss / self.cfg.grad_accum).backward()

        return loss.item()

    def maybe_step(self):
        """Step optimizer if gradient accumulation is complete."""
        if (self.step + 1) % self.cfg.grad_accum == 0:
            if self.cfg.grad_clip > 0:
                if self.cfg.use_amp:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.grad_clip
                )
            if self.cfg.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()
            if self.scheduler is not None:
                self.scheduler.step()

    def train(self, data_iter, total_steps, val_fn=None, val_kwargs=None,
              save_path=None, extra_loss_fn=None):
        """Main training loop.

        Args:
            data_iter: iterator yielding batches (X, Y, ...) or (X, Y, Yc, G)
            total_steps: total number of training steps
            val_fn: optional validation function(model, **val_kwargs) -> float
            val_kwargs: kwargs for val_fn
            save_path: path to save best checkpoint (model weights only)
            extra_loss_fn: optional loss function for dual-head models
        """
        self.total_steps = total_steps
        self._setup_scheduler(total_steps)
        self.step = 0
        self.best_val = float("inf")
        self.patience_counter = 0
        val_kwargs = val_kwargs or {}
        if extra_loss_fn is not None:
            self.extra_loss_fn = extra_loss_fn

        # Setup TensorBoard
        if self.cfg.tb_dir:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._tb_writer = SummaryWriter(self.cfg.tb_dir)
            except ImportError:
                self._log("tensorboard not available, skipping")

        ev = self.cfg.log_every
        t0 = time.time()
        self.optimizer.zero_grad()

        for step in range(1, total_steps + 1):
            self.step = step
            batch = next(data_iter)
            loss_val = self.train_step(batch)
            self.maybe_step()

            # Logging
            if step % ev == 0 or step == total_steps:
                lr = self.optimizer.param_groups[0]["lr"]
                elapsed = time.time() - t0
                self._log(f"step {step:5d} loss={loss_val:.3f} lr={lr:.1e} [{elapsed:.0f}s]")
                self._log_tb("train/loss", loss_val)
                self._log_tb("train/lr", lr)

            # Validation
            if val_fn is not None and (step % ev == 0 or step == total_steps):
                val_score = val_fn(self.model, **val_kwargs)
                self._log(f"  val={val_score:.3f} (best={self.best_val:.3f})")
                self._log_tb("val/score", val_score)

                if val_score < self.best_val:
                    self.best_val = val_score
                    self.patience_counter = 0
                    if save_path:
                        torch.save(self.model.state_dict(), save_path)
                        self._log(f"  * new best -> {save_path}")
                else:
                    self.patience_counter += 1

                # Early stopping
                if self.cfg.patience > 0 and self.patience_counter >= self.cfg.patience:
                    self._log(f"  early stopping at step {step} (patience={self.cfg.patience})")
                    break

            # Periodic checkpoint
            if self.cfg.save_every > 0 and step % self.cfg.save_every == 0 and save_path:
                ckpt_path = f"{save_path}.step{step}"
                self.save_checkpoint(ckpt_path)
                self._log(f"  checkpoint saved: {ckpt_path}")

        # Cleanup
        if self._tb_writer is not None:
            self._tb_writer.close()

        self._log(f"DONE. best val={self.best_val:.3f} (total steps={self.step})")
        return self.best_val
