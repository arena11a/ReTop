"""v6 M2 — HF packaging: `HMN3Config` / `HMNForCausalLM`.

Goal (docs/v6_scaling_roadmap.md M2): make HMN trainable with community
tooling (Trainer/SFTTrainer-class loops) and loadable with
save_pretrained/from_pretrained — no custom loop required.

Design notes (honest boundaries):
  * forward(labels=...) -> CausalLMOutput(loss=..., logits=gen_logits):
    the loss is the NATIVE per-target blend CE (exact logaddexp over the
    IRStats API) — identical math to hmn.recipe.loss_v33's blend term, no
    (B,T,V) blend tensor is built on this path.
  * forward() without labels -> logits = EXACT blended log-probs (B,T,V).
    This is the interop surface every HF LM exposes (an LM output head is
    irreducibly (B,T,V) — roadmap §M1-C note); the copy lane is scattered
    from the seed-group histograms. Generation via .generate() therefore
    sees the true dual-head distribution.
  * attention_mask is accepted and ignored: HMN's identity register treats
    every id as content; recipe batches already EOS-pad.
  * The research-grade trainer signals (Yc copy targets, G gate mask,
    seam anchors) stay native-side — pass them through `model.hmn` directly.
"""
import torch
from torch import nn

from transformers import GenerationConfig, PretrainedConfig, PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutput

from hmn.v3 import HMN3


class HMN3Config(PretrainedConfig):
    """Configuration for :class:`HMNForCausalLM` (mirrors HMN3.__init__)."""

    model_type = "hmn3"

    def __init__(
        self,
        vocab_size=3190,
        dim=96,
        state_dim=8,
        n_layers=3,
        n_experts=16,
        top_k=2,
        use_moe=False,
        use_think=False,
        k_max=4,
        tie_weights=True,
        gate_bias=0.0,
        asi_id=None,
        gate_mode="deterministic",
        user_id=None,
        stem_addr=False,
        seam_addr=False,
        max_run=16,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.dim = dim
        self.state_dim = state_dim
        self.n_layers = n_layers
        self.n_experts = n_experts
        self.top_k = top_k
        self.use_moe = use_moe
        self.use_think = use_think
        self.k_max = k_max
        self.tie_weights_flag = tie_weights      # avoid clashing with HF's own
        self.gate_bias = gate_bias
        self.asi_id = asi_id
        self.gate_mode = gate_mode
        self.user_id = user_id
        self.stem_addr = stem_addr
        self.seam_addr = seam_addr
        self.max_run = max_run
        # Aliases expected by HF GenerationMixin / cache infrastructure
        self.num_hidden_layers = n_layers


class HMNForCausalLM(GenerationMixin, PreTrainedModel):
    """Causal-LM wrapper around :class:`hmn.v3.HMN3` (see module docstring)."""

    config_class = HMN3Config
    base_model_prefix = "hmn"
    _supports_flash_attn = False

    def __init__(self, config):
        super().__init__(config)
        cfg = config
        self.hmn = HMN3(
            cfg.vocab_size, dim=cfg.dim, state_dim=cfg.state_dim,
            n_layers=cfg.n_layers, n_experts=cfg.n_experts, top_k=cfg.top_k,
            use_moe=cfg.use_moe, use_think=cfg.use_think, k_max=cfg.k_max,
            tie_weights=getattr(cfg, "tie_weights_flag", True),
            gate_bias=cfg.gate_bias, aux_copy=False, asi_id=cfg.asi_id,
            sparse_marginal=False, gate_mode=cfg.gate_mode,
            user_id=cfg.user_id, stem_addr=cfg.stem_addr,
            seam_addr=cfg.seam_addr, max_run=cfg.max_run,
        )
        # Explicitly set generation_config so top_k (an MoE model param) is
        # not leaked into the generation_config.json during save_pretrained.
        self.generation_config = GenerationConfig(do_sample=False)
        self.post_init()

    # -- embedding plumbing expected by HF utilities -------------------------
    def get_input_embeddings(self):
        return self.hmn.embed

    def set_input_embeddings(self, value):
        self.hmn.embed = value

    # -- core ----------------------------------------------------------------
    @torch.no_grad()
    def _blended_logprobs(self, out):
        """Exact blended log-probs (B,T,V) from the IRStats histograms.

        This is the interop surface: every HF LM returns (B,T,V) logits.
        The copy lane is scattered from the seed-group histograms built by
        IdentityRegister._index.  The cost is O(T * avg_payloads_per_group)
        writes — linear in the number of distinct (row, payload) pairs, NOT
        the forbidden O(T*V) per position.
        """
        st = out["stats"]
        B, T, V = out["gen_logits"].shape
        dev = out["gen_logits"].device
        g = out["g"].squeeze(-1)                        # (B,T)
        genp = out["gen_logits"]
        p_copy = torch.zeros(B, T, V, device=dev)
        if st.h_key.numel():
            bidx = torch.arange(B, device=dev).unsqueeze(1) \
                .expand(B, T).reshape(-1)
            key = bidx * st.G + st.col_grp.reshape(-1)
            order = torch.argsort(key, stable=True)
            sk = key[order]
            starts = torch.ones(sk.numel(), dtype=torch.long, device=dev)
            starts[1:] = sk[1:] != sk[:-1]
            bnd = starts.nonzero(as_tuple=True)[0]
            sizes = torch.diff(torch.cat([bnd, torch.tensor([sk.numel()],
                                                            device=dev)]))
            run_key = sk[bnd]
            lo = torch.searchsorted(st.h_key.contiguous(),
                                    (run_key * V).contiguous(), side="left")
            hi = torch.searchsorted(st.h_key.contiguous(),
                                    (run_key * V + V - 1).contiguous(),
                                    side="right")
            denom = st.denom.reshape(-1)[run_key].clamp(min=1.0)
            flat_p = p_copy.view(-1, V)
            for r in range(bnd.numel()):
                rows = order[bnd[r]:bnd[r] + sizes[r]]
                ys = st.h_y[lo[r]:hi[r]]
                if ys.numel() == 0:
                    continue
                frs = (st.h_cnt[lo[r]:hi[r]].float() / float(denom[r]))
                row_idx = rows[:, None].expand(-1, ys.numel()).reshape(-1)
                col_idx = ys[None, :].expand(rows.numel(), -1).reshape(-1)
                vals = frs[None, :].expand(rows.numel(), -1).reshape(-1)
                flat_p.index_put_((row_idx, col_idx), vals)
            del flat_p
        lp_c = p_copy.clamp(min=1e-12).log()
        gs = g.clamp(min=1e-12, max=1 - 1e-7)
        return torch.logaddexp(torch.log1p(-gs).unsqueeze(-1) + genp,
                               torch.log(gs).unsqueeze(-1) + lp_c)

    def forward(self, input_ids, attention_mask=None, labels=None,
                return_dict=True, **kwargs):
        out = self.hmn(input_ids)
        if labels is not None:
            st = out["stats"]
            genlp = out["gen_logits"]
            g = out["g"].squeeze(-1)
            lg_y = genlp.gather(2, labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
            pc = st.prob_at(labels)
            lp_c = torch.full_like(pc, float("-inf"))
            nz = pc > 0
            lp_c[nz] = pc[nz].log()
            l_one = torch.logaddexp(torch.log1p(-g.clamp(max=1 - 1e-7)) + lg_y,
                                    torch.log(g.clamp(min=1e-12)) + lp_c)
            ymask = labels != -100
            loss = -l_one[ymask].mean() if ymask.any() else genlp.new_zeros(())
            return CausalLMOutput(loss=loss, logits=out["gen_logits"])
        logits = self._blended_logprobs(out)
        return CausalLMOutput(loss=None, logits=logits)

    # -- generation plumbing --------------------------------------------------
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}

    def can_generate(self):
        return True


def to_hmn3_config(model: HMN3) -> HMN3Config:
    """Derive an HMN3Config from a native HMN3 instance."""
    dim = model.embed.weight.shape[1]
    return HMN3Config(
        vocab_size=model.ir.vocab, dim=dim,
        state_dim=model.blocks[0].F1.state_dim,
        n_layers=len(model.blocks),
        tie_weights=bool(torch.equal(model.dual.gen.weight[:, :dim],
                                     model.embed.weight)),
        gate_bias=float(model.dual.gate_bias),
        asi_id=model.ir.asi_id, user_id=model.ir.user_id,
        stem_addr=model.ir.stem_addr, seam_addr=model.seam_addr,
        max_run=model.seed_ptr.max_run if model.seed_ptr is not None else 16,
    )
