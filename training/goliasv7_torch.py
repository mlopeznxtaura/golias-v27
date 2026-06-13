"""
GoliasV7Torch: PyTorch nn.Module version of Golias-v7.
Loads warm-started numpy weights, trains with Unsloth acceleration.
Outputs v8 checkpoint.

Architecture (scaled from v6):
  Encoders:   m1(96->1536->384), m2(152->1536->384), m3(352->1536->384)
  Fusion:     1152->4 experts (top-2 gating), each 1152->1536->512
  Relation:   512->1536->512
  MLP pred:   512->1536->512
  Decode LM:  512->1536
  Attention:  2 blocks, d=512 (Unsloth-accelerated)
  c_comp:     384->1
  Meta:       H1(384), H2(384), TAU, K_t, step
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import os


# ═══════════════════════════════════════════════════════════
# Model Definition
# ═══════════════════════════════════════════════════════════

class Encoder(nn.Module):
    """Single encoder: input -> fc1 -> ReLU -> fc2 -> output."""
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden, bias=True)
        self.fc2 = nn.Linear(d_hidden, d_out, bias=True)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MoEExpert(nn.Module):
    """Single MoE expert: fc1 -> ReLU -> fc2."""
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden, bias=True)
        self.fc2 = nn.Linear(d_hidden, d_out, bias=True)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MoEFusion(nn.Module):
    """Mixture of Experts fusion with top-k gating."""
    def __init__(self, d_in, d_hidden, d_out, num_experts=4, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_in, num_experts, bias=True)
        self.experts = nn.ModuleList([
            MoEExpert(d_in, d_hidden, d_out) for _ in range(num_experts)
        ])

    def forward(self, x):
        # x: (batch, d_in)
        logits = self.gate(x)  # (batch, num_experts)
        weights = F.softmax(logits, dim=-1)

        # Top-k gating
        topk_weights, topk_indices = torch.topk(weights, self.top_k, dim=-1)
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-8)

        # Sparse MoE: compute only top-k experts
        batch_size = x.shape[0]
        d_out = self.experts[0].fc2.out_features
        out = torch.zeros(batch_size, d_out, device=x.device, dtype=x.dtype)

        for k in range(self.top_k):
            expert_idx = topk_indices[:, k]  # (batch,)
            weight = topk_weights[:, k]  # (batch,)

            # Group samples by expert
            for e in range(self.num_experts):
                mask = (expert_idx == e)
                if mask.any():
                    expert_out = self.experts[e](x[mask])
                    out[mask] += weight[mask].unsqueeze(-1) * expert_out

        return out


class AttentionBlock(nn.Module):
    """Attention-style residual block for single-vector processing.
    Uses multi-head linear projections (degenerate attention for single token).
    Compatible with v6 weight layout: (d_in, d_out) for WQ/WK/WV/WO.
    """
    def __init__(self, d_model=512, n_heads=8):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.WQ = nn.Linear(d_model, d_model, bias=True)
        self.WK = nn.Linear(d_model, d_model, bias=True)
        self.WV = nn.Linear(d_model, d_model, bias=True)
        self.WO = nn.Linear(d_model, d_model, bias=True)
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        # x: (batch, d_model) - single vector per sample
        B, D = x.shape
        
        # For single-vector input, attention degenerates to gated linear projection.
        # Compute query, key, value projections
        q = self.WQ(x)  # (B, D)
        k = self.WK(x)  # (B, D)
        v = self.WV(x)  # (B, D)
        
        # Scale and gate
        scale = self.temperature / (self.d_head ** 0.5)
        # Compute per-element attention (simplified for single token)
        gate = torch.sigmoid((q * k).sum(dim=-1, keepdim=True) * scale)  # (B, 1)
        out = gate * v  # (B, D)
        
        out = self.WO(out)
        return out


class GoliasV7Torch(nn.Module):
    """
    Golias-Γ-v7: Scaled MoE Fusion (PyTorch version).
    ~18.97M params, warm-started from v6.
    """
    def __init__(self):
        super().__init__()

        # Dimensions
        d_m1_in, d_m2_in, d_m3_in = 96, 152, 352
        d_enc_hidden, d_enc_out = 1536, 384
        d_fusion_in = d_enc_out * 3  # 1152
        d_fusion_hidden, d_fusion_out = 1536, 512
        num_experts, top_k = 4, 2
        d_attn, n_attn_blocks, n_heads = 512, 2, 8
        d_rel_hidden, d_mlp_hidden = 1536, 1536
        d_decode = 1536
        d_meta = 384

        # Encoders
        self.m1_enc = Encoder(d_m1_in, d_enc_hidden, d_enc_out)
        self.m2_enc = Encoder(d_m2_in, d_enc_hidden, d_enc_out)
        self.m3_meta = Encoder(d_m3_in, d_enc_hidden, d_enc_out)

        # MoE Fusion
        self.fusion = MoEFusion(d_fusion_in, d_fusion_hidden, d_fusion_out,
                                num_experts, top_k)

        # Relation
        self.relation = nn.Sequential(
            nn.Linear(d_fusion_out, d_rel_hidden, bias=True),
            nn.ReLU(),
            nn.Linear(d_rel_hidden, d_fusion_out, bias=True),
        )

        # MLP Predictor
        self.mlp_pred = nn.Sequential(
            nn.Linear(d_fusion_out, d_mlp_hidden, bias=True),
            nn.ReLU(),
            nn.Linear(d_mlp_hidden, d_fusion_out, bias=True),
        )

        # Decode LM
        self.decode_lm = nn.Linear(d_fusion_out, d_decode, bias=True)

        # Attention blocks
        self.attn_blocks = nn.ModuleList([
            AttentionBlock(d_attn, n_heads) for _ in range(n_attn_blocks)
        ])

        # Concept compression
        self.c_comp = nn.Linear(d_meta, 1, bias=True)

        # Meta state (non-trainable buffers)
        self.register_buffer('H1_prev', torch.zeros(d_meta))
        self.register_buffer('H2_prev', torch.zeros(d_meta))
        self.register_buffer('K_t', torch.tensor(1.0))
        self.register_buffer('TAU', torch.tensor(0.28))
        self.register_buffer('step', torch.tensor(0.0))

        self._dims = {
            'm1_in': d_m1_in, 'm2_in': d_m2_in, 'm3_in': d_m3_in,
            'enc_hidden': d_enc_hidden, 'enc_out': d_enc_out,
            'fusion_in': d_fusion_in, 'fusion_hidden': d_fusion_hidden,
            'fusion_out': d_fusion_out, 'num_experts': num_experts,
            'd_attn': d_attn, 'n_attn_blocks': n_attn_blocks,
            'd_decode': d_decode, 'd_meta': d_meta,
        }

    def forward(self, m1_input, m2_input, m3_input, meta_state=None):
        """
        Args:
            m1_input: (batch, 96)  - text/language features
            m2_input: (batch, 152) - geometry + pose/intrinsic
            m3_input: (batch, 352) - meta/context features
            meta_state: optional dict with H1_prev, H2_prev, TAU, K_t

        Returns:
            dict with 'pred', 'decode', 'relation_out', 'meta'
        """
        # Encode
        e1 = self.m1_enc(m1_input)  # (B, 384)
        e2 = self.m2_enc(m2_input)  # (B, 384)
        e3 = self.m3_meta(m3_input)  # (B, 384)

        # Concatenate for fusion
        fused_in = torch.cat([e1, e2, e3], dim=-1)  # (B, 1152)

        # MoE Fusion
        fused_out = self.fusion(fused_in)  # (B, 512)

        # Relation
        rel_out = self.relation(fused_out)  # (B, 512)

        # MLP prediction
        pred = self.mlp_pred(rel_out)  # (B, 512)

        # Decode
        decode = self.decode_lm(pred)  # (B, 1536)

        # Attention (2 blocks)
        attn_out = pred
        for attn_block in self.attn_blocks:
            attn_out = attn_out + attn_block(attn_out)  # residual

        # Concept compression (from encoder outputs)
        concepts = torch.cat([e1, e2, e3], dim=-1)  # (B, 1152)
        # Use first 384 dims for c_comp (matching d_meta)
        comp_score = self.c_comp(concepts[:, :self._dims['d_meta']])  # (B, 1)

        # Meta update (simplified for training)
        if meta_state is not None:
            self.H1_prev = meta_state.get('H1_prev', self.H1_prev)
            self.H2_prev = meta_state.get('H2_prev', self.H2_prev)
            self.TAU = meta_state.get('TAU', self.TAU)
            self.K_t = meta_state.get('K_t', self.K_t)

        return {
            'pred': pred,
            'decode': decode,
            'relation_out': rel_out,
            'attn_out': attn_out,
            'comp_score': comp_score,
            'encoder_outputs': (e1, e2, e3),
            'fusion_out': fused_out,
        }

    @property
    def total_params(self):
        return sum(p.numel() for p in self.parameters())

    def load_warmstart(self, path):
        """Load warm-started numpy weights into this PyTorch module."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        sd = data['state_dict']

        def set_weight(module, key, arr):
            """Set weight from numpy array, handling transpose for Linear layers."""
            if isinstance(module, nn.Linear):
                # nn.Linear weight is (out, in), our numpy is (in, out)
                if 'weight' in key:
                    arr = arr.T.copy()
            
            # Try via state_dict first
            mod_sd = module.state_dict(keep_vars=True)
            if key in mod_sd:
                target = mod_sd[key]
                t = torch.from_numpy(arr).float().to(target.device)
                if target.shape != t.shape:
                    target.data = t
                else:
                    target.data.copy_(t)
                return
            
            # Fallback: named_parameters / named_buffers
            for name, p in list(module.named_parameters()) + list(module.named_buffers()):
                if name == key:
                    t = torch.from_numpy(arr).float().to(p.device)
                    if p.shape != t.shape:
                        p.data = t
                    else:
                        p.data.copy_(t)
                    return
            print(f"  WARN: could not find {key} in module")

        # Encoders
        for enc_name, module in [('m1_enc', self.m1_enc), ('m2_enc', self.m2_enc),
                                  ('m3_meta', self.m3_meta)]:
            for layer in ['fc1', 'fc2']:
                for param in ['weight', 'bias']:
                    v7_key = f'{enc_name}.{layer}.W' if param == 'weight' else f'{enc_name}.{layer}.b'
                    arr = sd.get(v7_key)
                    if arr is not None:
                        target = getattr(module, layer)
                        set_weight(target, param, arr)

        # Fusion
        for i in range(4):
            for layer in ['fc1', 'fc2']:
                for param in ['weight', 'bias']:
                    v7_key = f'fusion.expert{i}.{layer}.W' if param == 'weight' else f'fusion.expert{i}.{layer}.b'
                    arr = sd.get(v7_key)
                    if arr is not None:
                        target = getattr(self.fusion.experts[i], layer)
                        set_weight(target, param, arr)

        # Fusion gate
        gate_arr = sd.get('fusion.gate.W')
        if gate_arr is not None:
            set_weight(self.fusion.gate, 'weight', gate_arr)
        gate_b = sd.get('fusion.gate.b')
        if gate_b is not None:
            set_weight(self.fusion.gate, 'bias', gate_b)

        # Relation
        for layer, idx in [('fc1', 0), ('fc2', 2)]:
            for param in ['weight', 'bias']:
                v7_key = f'relation.{layer}.W' if param == 'weight' else f'relation.{layer}.b'
                arr = sd.get(v7_key)
                if arr is not None:
                    target = self.relation[idx]
                    set_weight(target, param, arr)

        # MLP Pred
        for layer, idx in [('fc1', 0), ('fc2', 2)]:
            for param in ['weight', 'bias']:
                v7_key = f'mlp_pred.{layer}.W' if param == 'weight' else f'mlp_pred.{layer}.b'
                arr = sd.get(v7_key)
                if arr is not None:
                    target = self.mlp_pred[idx]
                    set_weight(target, param, arr)

        # Decode LM
        for param in ['weight', 'bias']:
            v7_key = f'decode_lm.W' if param == 'weight' else f'decode_lm.b'
            arr = sd.get(v7_key)
            if arr is not None:
                set_weight(self.decode_lm, param, arr)

        # Attention blocks
        for b in range(2):
            for proj in ['WQ', 'WK', 'WV', 'WO']:
                v7_key = f'attn.block{b}.{proj}'
                arr = sd.get(v7_key)
                if arr is not None:
                    target = getattr(self.attn_blocks[b], proj)
                    set_weight(target, 'weight', arr)
            t_key = f'attn.block{b}._t'
            t_arr = sd.get(t_key)
            if t_arr is not None:
                self.attn_blocks[b].temperature.data.fill_(float(t_arr.item() if hasattr(t_arr, 'item') else t_arr))

        # c_comp
        for param in ['weight', 'bias']:
            v7_key = f'c_comp.W' if param == 'weight' else f'c_comp.b'
            arr = sd.get(v7_key)
            if arr is not None:
                set_weight(self.c_comp, param, arr)

        # Meta
        for buf_name, key in [('H1_prev', 'meta.H1_prev'), ('H2_prev', 'meta.H2_prev'),
                               ('TAU', 'meta.TAU'), ('K_t', 'meta.K_t'),
                               ('step', 'meta.step')]:
            arr = sd.get(key)
            if arr is not None:
                buf = getattr(self, buf_name)
                val = torch.from_numpy(arr).float()
                if buf.data.shape != val.shape:
                    buf.data = val.to(buf.device)
                else:
                    buf.data.copy_(val.to(buf.device))

        print(f"Loaded warm-started weights from {path}")

    def load_checkpoint(self, path):
        """Load a v7/v8 pickle checkpoint (same .W/.b key layout)."""
        self.load_warmstart(path)


if __name__ == '__main__':
    model = GoliasV7Torch()
    print(f"GoliasV7Torch initialized: {model.total_params:,} parameters")
    print(f"  Encoders: m1({model._dims['m1_in']}->{model._dims['enc_out']}), "
          f"m2({model._dims['m2_in']}->{model._dims['enc_out']}), "
          f"m3({model._dims['m3_in']}->{model._dims['enc_out']})")
    print(f"  Fusion: {model._dims['fusion_in']}->{model._dims['fusion_out']} "
          f"({model._dims['num_experts']} experts, top-2)")
    print(f"  Attention: {model._dims['n_attn_blocks']} blocks x d={model._dims['d_attn']}")
    print(f"  Meta: d={model._dims['d_meta']}")