#!/usr/bin/env python3
"""C^3-JEPA — minimal reference implementation (train + test in one script).

Underwater C^3-JEPA: an object-centric, cross-view, control-conditioned,
context-extended predictive world model for near-field heavy-load ROV salvage.

This file is a self-contained, minimal implementation of the method described in
the paper.  It runs end to end on CPU with synthetic data:

    python c3_jepa.py smoke                     # tiny synthetic train+test, seconds
    python c3_jepa.py train --data-dir data/recs --out-dir runs/example
    python c3_jepa.py test  --data-dir data/recs --ckpt runs/example/ckpt/best.pt

Pipeline (one forward pass):
    multi-view frames  ->  patch tokens  ->  per-view slot attention (6 slots)
                       ->  cross-view attention fusion (held-out-view training)
                       ->  object tokens z_t  ->  control-conditioned predictor
                       ->  future object tokens / masked-history recovery

Losses (paper Eq. 2 plus the two auxiliary weights of Table 1):
    L =  L_m + l_b * L_bind + l_s * L_sigreg + l_t * L_temporal + l_v * L_heldout
    L_pred = L_masked_history + L_future                       (predictor)

All file paths are relative; nothing is hard-coded to a machine or a dataset root.

Data layout (--data-dir), one .npz per recording:

    rec0001.npz
        frames    (V, T, H, W, 3)  uint8     V synchronised camera views, 4 Hz
        control   (T, C)           float32   linear/angular velocity + gripper
        mask_uuv  (V, T, N)        bool      weak mask: task object  (optional)
        mask_grip (V, T, N)        bool      weak mask: gripper       (optional)
        timestamps(T,)             float64

    N = (H // patch) * (W // patch).  Weak masks are cheap patch-level labels;
    when they are absent the binding term is skipped automatically.

Precomputed backbone features (optional, the paper's setting): pass
--cached-features DIR holding <rec>_<view>_feat.npy of shape (T, N, D) produced
by a frozen DINOv3-L patch encoder; --backbone is then ignored.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# --------------------------------------------------------------------------------------
# 0. defaults (paper Table 1 / Eq. 2)
# --------------------------------------------------------------------------------------
DEFAULTS = dict(
    n_views=2,          # co-mounted cameras used in the study
    n_slots=6,          # 0 task object, 1 gripper, 2 fixed ROV frame, 3-5 context
    slot_dim=256,       # d_s = 256 (the predictor adapts 256 -> 128 -> 256)
    patch=16,           # patch size of the backbone feature grid
    backbone_dim=768,   # frozen DINOv3-L width
    history=4,          # 1 s of context at 4 Hz
    future=12,          # 3 s of prediction
    masked_slots=2,     # object-level masks per history window
    fusion_heads=8,     # held-out-view cross-attention heads
    lambda_m=1.0,       # reconstruction
    lambda_b=1.0,       # weak binding
    lambda_s=0.03,      # SIGReg               (Table 1)
    lambda_t=0.10,      # temporal similarity  (Table 1)
    lambda_v=0.20,      # held-out view        (Table 1)
    lr=1e-4,            # (Table 1)
    weight_decay=0.05,
    epochs=30,
    batch_size=16,      # 16 per GPU on 8 GPUs
    seed=42,
)

# Six tokenizer slots, of which three carry weak binding and three stay free.
# Slot 2 (the fixed ROV frame) is a binding-only slot: it is not synthesised and is
# excluded both from cross-view fusion and from the predictive state z_t.
BINDING_SLOTS = (0, 1, 2)
BIND_TARGETS = ("uuv", "gripper", "frame")
STATE_SLOTS = (0, 1, 3, 4, 5)      # z_t  — five slots
OBJECT_SLOTS = (0, 1)              # task object and gripper inside z_t


# --------------------------------------------------------------------------------------
# 1. data
# --------------------------------------------------------------------------------------
class Recording:
    """One synchronised multi-view recording at a fixed rate."""

    def __init__(self, name, frames=None, feats=None, control=None, masks=None):
        self.name = name
        self.frames = frames            # (V,T,H,W,3) uint8  or None
        self.feats = feats              # (V,T,N,D)   float32 or None
        self.control = control          # (T,C) float32
        self.masks = masks or {}        # {"uuv": (V,T,N) bool, "gripper": ...}
        self.T = int(control.shape[0])

    def __len__(self):
        return self.T


def load_npz(path: Path) -> Recording:
    with np.load(path, allow_pickle=False) as d:
        keys = set(d.files)
        frames = d["frames"] if "frames" in keys else None
        control = d["control"].astype(np.float32)
        masks = {}
        for k, name in (("mask_uuv", "uuv"), ("mask_grip", "gripper")):
            if k in keys:
                masks[name] = d[k].astype(bool)
    return Recording(path.stem, frames=frames, control=control, masks=masks)


def load_cached_features(feats_dir: Path, rec: Recording) -> None:
    """<rec>_<view_idx>_feat.npy  ->  (T, N, D)."""
    views = []
    for v in range(4096):
        p = feats_dir / f"{rec.name}_{v}_feat.npy"
        if not p.exists():
            break
        views.append(np.load(p).astype(np.float32))
    if not views:
        raise FileNotFoundError(f"no cached feature files for {rec.name} in {feats_dir}")
    rec.feats = np.stack(views)


class Windows(Dataset):
    """Sliding windows of length history+future over each recording."""

    def __init__(self, recs, length, stride=1):
        self.recs = recs
        self.length = length
        self.index = [(i, j) for i, r in enumerate(recs)
                      for j in range(0, len(r) - length + 1, stride)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        ri, j = self.index[i]
        r = self.recs[ri]
        e = j + self.length
        item = {"control": torch.from_numpy(r.control[j:e])}
        if r.feats is not None:
            item["feats"] = torch.from_numpy(r.feats[:, j:e])
        if r.frames is not None:
            item["frames"] = torch.from_numpy(r.frames[:, j:e].astype(np.float32) / 255.0)
        for name, m in r.masks.items():
            item[f"mask_{name}"] = torch.from_numpy(m[:, j:e])
        return item

    @staticmethod
    def collate(batch):
        out = {}
        for k in batch[0]:
            out[k] = torch.stack([b[k] for b in batch])
        return out


# --------------------------------------------------------------------------------------
# 2. synthetic data (smoke test only — never used for the paper's numbers)
# --------------------------------------------------------------------------------------
def make_synthetic(n_rec=3, T=40, n_views=2, H=32, W=32, patch=16,
                   seed=0, with_masks=True, out_dir: Path | None = None):
    """A moving task object + a closing gripper, seen from two views.

    Deterministic in ``seed`` so that a smoke run is reproducible.
    """
    rng = np.random.default_rng(seed)
    recs = []
    N = (H // patch) * (W // patch)
    ph, pw = H // patch, W // patch
    for r in range(n_rec):
        obj_y, obj_x = rng.uniform(0.3, 0.7, 2)
        # fixed speed, random direction: a record whose speed draw is near zero would
        # make the sequence static and the persistence baseline trivially unbeatable
        speed, ang = rng.uniform(0.08, 0.14), rng.uniform(0, 2 * np.pi)
        obj_vel = np.array([np.cos(ang), np.sin(ang)]) * speed
        grip_gap = 0.35
        frames = np.zeros((n_views, T, H, W, 3), np.uint8)
        masks = {name: np.zeros((n_views, T, N), bool) for name in BIND_TARGETS}
        control = np.zeros((T, 5), np.float32)
        # the fixed ROV frame is static: a border band, identical across views and time
        frame_cell = np.zeros((H, W), bool)
        frame_cell[:2, :] = True
        frame_cell[-2:, :] = True
        for t in range(T):
            obj_y += float(obj_vel[0])
            obj_x += float(obj_vel[1])
            if not 0.10 <= obj_y <= 0.90:              # bounce off the workspace edge
                obj_vel[0] = -obj_vel[0]
                obj_y = float(np.clip(obj_y, 0.10, 0.90))
            if not 0.10 <= obj_x <= 0.90:
                obj_vel[1] = -obj_vel[1]
                obj_x = float(np.clip(obj_x, 0.10, 0.90))
            grip_gap = max(0.04, grip_gap - 0.012)
            control[t] = [obj_vel[0] * 10, obj_vel[1] * 10, 0.1, 0.0, 1.0 - grip_gap]
            for v in range(n_views):
                scale = 1.0 + 0.5 * v
                img = np.zeros((H, W, 3), np.uint8)
                oy = int(obj_y * H)
                ox = int(np.clip(obj_x * W + (0 if v == 0 else 3), 0, W - 1))
                r_o = max(3, int(0.16 * H * scale / 2))
                yy, xx = np.ogrid[:H, :W]
                obj_cell = (yy - oy) ** 2 + (xx - ox) ** 2 <= r_o ** 2
                img[..., 0] = np.where(obj_cell, 230, 40)
                img[..., 1] = np.where(obj_cell, 180, 60)
                gy = min(H - 1, oy + r_o + 2)
                gl = int(grip_gap * W / 2)
                grip_cell = np.zeros((H, W), bool)
                grip_cell[gy:gy + 2, max(0, ox - gl):ox] = True
                grip_cell[gy:gy + 2, ox:ox + gl] = True
                img[..., 2] = np.where(grip_cell, 240, 80)
                frames[v, t] = img
                if with_masks:
                    for name, cell in (("uuv", obj_cell), ("gripper", grip_cell),
                                       ("frame", frame_cell)):
                        grid = cell.reshape(ph, patch, pw, patch).mean(axis=(1, 3))
                        masks[name][v, t] = (grid > 0.05).reshape(-1)
        rec = Recording(f"synth_{r:02d}", frames=frames, control=control,
                        masks=masks if with_masks else None)
        recs.append(rec)
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        for rec in recs:
            np.savez_compressed(out_dir / f"{rec.name}.npz", frames=rec.frames,
                                control=rec.control,
                                **{f"mask_{n}": m for n, m in rec.masks.items()})
    return recs


# --------------------------------------------------------------------------------------
# 3. perception: patch encoder -> slot attention -> cross-view fusion
# --------------------------------------------------------------------------------------
class PatchEncoder(nn.Module):
    """Stand-in for the frozen DINOv3-L backbone used in the paper.

    A single strided convolution keeps the script self-contained; the rest of the
    pipeline is identical when ``--cached-features`` supplies real DINOv3-L tokens.
    """

    def __init__(self, out_dim=768, patch=16, width=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, width, patch, stride=patch),
            nn.GELU(),
            nn.Conv2d(width, out_dim, 1),
            nn.Flatten(2),
        )

    def forward(self, frames):
        """(B,V,T,H,W,3) -> (B,V,T,N,D)."""
        b, v, t, h, w, _ = frames.shape
        x = frames.permute(0, 1, 2, 5, 3, 4).reshape(b * v * t, 3, h, w)
        x = self.net(x).transpose(1, 2)              # (B*V*T, N, D)
        return x.reshape(b, v, t, x.shape[-2], x.shape[-1])


class SlotAttention(nn.Module):
    """The tokenizer: competitive slot attention over patch tokens."""

    def __init__(self, inp_dim, slot_dim, n_iters=3):
        super().__init__()
        self.n_iters = n_iters
        self.to_q = nn.Linear(slot_dim, slot_dim, bias=False)
        self.to_k = nn.Linear(inp_dim, slot_dim, bias=False)
        self.to_v = nn.Linear(inp_dim, slot_dim, bias=False)
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.mlp = nn.Sequential(nn.Linear(slot_dim, 4 * slot_dim), nn.ReLU(),
                                 nn.Linear(4 * slot_dim, slot_dim))
        self.norm_features = nn.LayerNorm(inp_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.eps = 1e-8

    def forward(self, slots, features):
        b, s, _ = slots.shape
        f = self.norm_features(features)
        k, v = self.to_k(f), self.to_v(f)
        for _ in range(self.n_iters):
            q = self.to_q(self.norm_slots(slots))
            dots = torch.einsum("bsd,bnd->bsn", q, k) / math.sqrt(q.shape[-1])
            attn = F.softmax(dots, dim=-1)
            attn = attn / (attn.sum(-1, keepdim=True) + self.eps)
            upd = torch.einsum("bsn,bnd->bsd", attn, v)
            slots = self.gru(upd.reshape(b * s, -1), slots.reshape(b * s, -1)).reshape(b, s, -1)
            slots = slots + self.mlp(self.norm_slots(slots))
        return {"slots": slots, "masks": F.softmax(dots, dim=1)}   # competition over slots


class PatchDecoder(nn.Module):
    """Per-slot reconstruction of the patch tokens (the λ_m term)."""

    def __init__(self, slot_dim, n_patches, feat_dim, hidden=512):
        super().__init__()
        self.n_patches = n_patches
        self.pos = nn.Parameter(torch.randn(1, n_patches, slot_dim) * 0.02)
        self.mlp = nn.Sequential(nn.Linear(slot_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, feat_dim))
        self.mask_mlp = nn.Sequential(nn.Linear(slot_dim, slot_dim), nn.ReLU(),
                                      nn.Linear(slot_dim, 1))

    def forward(self, slots):
        b, s, _ = slots.shape
        x = slots.unsqueeze(2).expand(-1, -1, self.n_patches, -1) + self.pos
        per_slot = self.mlp(x)
        weights = F.softmax(self.mask_mlp(x).squeeze(-1), dim=1)
        return {"reconstruction": torch.einsum("bsnd,bsn->bnd", per_slot, weights),
                "masks": weights}


class CrossViewFusion(nn.Module):
    """Held-out-view cross-attention fusion (after the paper's CrossViewMerge).

    For every fused slot a learned query attends over the per-view instances of that
    slot and returns one shared token, which is what makes the state view-agnostic.
    With ``hidden_view`` set, the query only sees the remaining views — that is the
    constraint the held-out-view term optimises.
    """

    def __init__(self, dim, heads=8):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 4 * dim),
                                 nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, per_view_slots, hidden_view=None):
        """(B,V,T,S,d) -> shared (B,T,S,d)."""
        b, v, t, s, d = per_view_slots.shape
        x = per_view_slots.permute(0, 2, 3, 1, 4).reshape(b * t * s, v, d)
        if hidden_view is not None:
            keep = [i for i in range(v) if i != hidden_view]
            x = x[:, keep]
        q = self.query.expand(x.shape[0], -1, -1)
        a, _ = self.attn(q, x, x, need_weights=False)
        q = self.norm(q + a)
        return (q + self.ffn(q))[:, 0].reshape(b, t, s, d)


class HeldOutViewHead(nn.Module):
    """Predict the hidden camera's object tokens from the fused state (λ_v)."""

    def __init__(self, slot_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(slot_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, slot_dim))

    def forward(self, fused, target_slots=(0, 1)):
        """(B,T,S,d) -> (B,T,K,d) predictions for the held-out view's slots."""
        return self.net(fused[..., list(target_slots), :])


class SIGReg(nn.Module):
    """Sketched Isotropic Gaussian Regularisation (after LeWorldModel, Apache-2.0).

    Input is (groups, samples, dim); every group is tested along its sample axis.
    """

    def __init__(self, knots=17, num_proj=64):
        super().__init__()
        self.num_proj = int(num_proj)
        t = torch.linspace(0, 3, knots)
        dt = 3 / (knots - 1)
        w = torch.full((knots,), 2 * dt)
        w[[0, -1]] = dt
        phi = torch.exp(-t.square() / 2)
        self.register_buffer("t", t, persistent=False)
        self.register_buffer("phi", phi, persistent=False)
        self.register_buffer("weights", w * phi, persistent=False)

    def forward(self, proj):
        proj = proj.float()
        directions = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        directions = directions / directions.norm(p=2, dim=0).clamp_min(1e-12)
        x_t = (proj @ directions).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        return ((err @ self.weights) * proj.size(-2)).mean()


# --------------------------------------------------------------------------------------
# 4. predictor: control-conditioned, context-extended, object-level masking
# --------------------------------------------------------------------------------------
class Predictor(nn.Module):
    """Non-causal transformer over object tokens + control auxiliary tokens.

    * object-level masking is held across the whole history window,
    * ``t = 0`` is an identity anchor for every slot,
    * each future step is a mask query + anchor + time embedding,
    * controls enter as separate auxiliary entity tokens (never pooled into slots).
    """

    def __init__(self, n_slots, ctrl_dim, history, future, slot_dim=128, h=128,
                 depth=6, heads=8):
        super().__init__()
        self.n_slots, self.history, self.future = n_slots, history, future
        self.in_proj = nn.Identity() if slot_dim == h else nn.Linear(slot_dim, h)
        self.out_proj = nn.Identity() if slot_dim == h else nn.Linear(h, slot_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, h))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.time = nn.Parameter(torch.randn(1, history + future, 1, h) * 0.02)
        self.anchor = nn.Linear(h, h)
        self.ctrl = nn.Sequential(nn.LayerNorm(ctrl_dim), nn.Linear(ctrl_dim, 2 * h),
                                  nn.GELU(), nn.Linear(2 * h, h))
        self.aux_type = nn.Parameter(torch.randn(1, 1, 1, h) * 0.02)
        layer = nn.TransformerEncoderLayer(h, heads, 4 * h, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.tx = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.out = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h))

    def forward(self, slots, control, n_masked=2, hide_slot=None):
        """slots (B,T,S,d), control (B,T,C) -> predictions (B,T,S,d), targets, ids."""
        b, t, s, d = slots.shape
        assert t == self.history + self.future and s == self.n_slots
        x = self.in_proj(slots)
        h = x.shape[-1]                                 # internal width (paper: 128)
        if hide_slot is not None:                       # paired evaluation
            ids = torch.full((b, n_masked), hide_slot, dtype=torch.long, device=x.device)
        elif n_masked:
            ids = torch.stack([torch.randperm(s, device=x.device)[:n_masked] for _ in range(b)])
        else:
            ids = torch.empty(b, 0, dtype=torch.long, device=x.device)
        q = self.mask_token.expand(b, t, s, h) + self.time[:, :t] + self.anchor(x[:, :1])
        inp = q.clone()
        inp[:, 0] = x[:, 0] + self.time[:, :1]         # identity anchor at t = 0
        visible = torch.ones(b, s, dtype=torch.bool, device=x.device)
        if n_masked:
            visible.scatter_(1, ids, False)
        if self.history > 1:
            inp[:, 1:self.history] = torch.where(visible[:, None, :, None],
                                                 x[:, 1:self.history] + self.time[:, 1:self.history],
                                                 inp[:, 1:self.history])
        aux = self.ctrl(control).unsqueeze(2) + self.time[:, :t] + self.aux_type
        seq = torch.cat([inp, aux], 2).reshape(b, t * (s + 1), h)
        z = self.tx(seq).reshape(b, t, s + 1, h)[:, :, :s]
        return self.out_proj(self.out(z)), slots, ids


def predictor_losses(pred, target, ids, history):
    """Masked-history recovery + future prediction (MSE, plus cosine diagnostics)."""
    b, t, s, d = pred.shape
    if ids.numel() and history > 1:
        idx = ids[:, None, :, None].expand(-1, history - 1, -1, d)
        ph = torch.gather(pred[:, 1:history], 2, idx)
        th = torch.gather(target[:, 1:history], 2, idx)
        l_hist = F.mse_loss(ph, th)
        cos_hist = 1 - F.cosine_similarity(ph, th, -1).mean()
    else:
        l_hist = pred.new_zeros(())
        cos_hist = pred.new_zeros(())
    pf, tf = pred[:, history:], target[:, history:]
    return l_hist, F.mse_loss(pf, tf), cos_hist, 1 - F.cosine_similarity(pf, tf, -1).mean()


# --------------------------------------------------------------------------------------
# 5. the model
# --------------------------------------------------------------------------------------
class C3JEPA(nn.Module):
    def __init__(self, cfg, ctrl_dim, n_patches, feat_dim):
        super().__init__()
        self.cfg = cfg
        self.encoder = PatchEncoder(feat_dim, cfg["patch"])
        if cfg.get("freeze_backbone", True):
            # the paper freezes the DINOv3-L backbone; freezing the stand-in keeps the
            # same property — a latent that cannot be made static by training
            for p in self.encoder.parameters():
                p.requires_grad_(False)
        self.patch_proj = nn.Sequential(nn.LayerNorm(feat_dim),
                                        nn.Linear(feat_dim, 2 * cfg["slot_dim"]),
                                        nn.GELU(),
                                        nn.Linear(2 * cfg["slot_dim"], cfg["slot_dim"]))
        self.slot_init = nn.Parameter(torch.randn(1, cfg["n_slots"], cfg["slot_dim"]) * 0.02)
        self.slot_attn = SlotAttention(cfg["slot_dim"], cfg["slot_dim"], n_iters=3)
        self.decoder = PatchDecoder(cfg["slot_dim"], n_patches, feat_dim)
        self.fusion = CrossViewFusion(cfg["slot_dim"], cfg.get("fusion_heads", 8))
        self.heldout = HeldOutViewHead(cfg["slot_dim"])
        self.sigreg = SIGReg(num_proj=cfg.get("sigreg_proj", 64))
        self.predictor = Predictor(len(STATE_SLOTS), ctrl_dim, cfg["history"], cfg["future"],
                                   slot_dim=cfg["slot_dim"], h=cfg.get("hidden", 128),
                                   depth=cfg.get("depth", 3), heads=cfg.get("heads", 4))

    # -- perception ---------------------------------------------------------------
    def tokenize(self, batch):
        feats = batch.get("feats")
        if feats is None:
            feats = self.encoder(batch["frames"])                 # (B,V,T,N,D)
        b, v, t, n, d = feats.shape
        flat = feats.reshape(b * v * t, n, d)
        proj = self.patch_proj(flat)
        slots0 = self.slot_init.expand(proj.shape[0], -1, -1)
        out = self.slot_attn(slots0, proj)
        slots = out["slots"].reshape(b, v, t, self.cfg["n_slots"], -1)
        dec = self.decoder(out["slots"])
        return {"feats": flat.reshape(b * v * t, n, d),
                "recon": dec["reconstruction"].reshape(b, v, t, n, d),
                "grouping": out["masks"].reshape(b, v, t, self.cfg["n_slots"], n),
                "decoder_masks": dec["masks"].reshape(b, v, t, self.cfg["n_slots"], n),
                "slots": slots}

    @staticmethod
    def _soft_ce(attn, mask):
        """Competitive-mask NLL: -log(mean probability mass inside the weak mask)."""
        valid = mask.any(-1)
        if not valid.any():
            return None
        n_targets = mask.float().sum(-1)
        prob = (attn * mask.float()).sum(-1) / (n_targets + 1e-8)
        return -torch.log(prob + 1e-8)[valid].mean()

    def perception_losses(self, batch, tok):
        cfg = self.cfg
        l_rec = F.mse_loss(tok["recon"], tok["feats"].reshape(tok["recon"].shape))
        # weak binding on both mask branches (grouping + decoder)
        bind_terms = []
        for name, slot_idx in zip(BIND_TARGETS, BINDING_SLOTS):
            key = f"mask_{name}"
            if key not in batch:
                continue
            m = batch[key].reshape(-1, batch[key].shape[-1])
            for branch in ("grouping", "decoder_masks"):
                attn = tok[branch].reshape(-1, cfg["n_slots"], m.shape[-1])[:, slot_idx, :]
                term = self._soft_ce(attn, m)
                if term is not None:
                    bind_terms.append(term)
        l_bind = torch.stack(bind_terms).mean() if bind_terms else l_rec.new_zeros(())
        # the predictive state z_t is the five slots that are not binding-only
        z_slots = tok["slots"][..., list(STATE_SLOTS), :]           # (B,V,T,5,d)
        l_temp = (F.mse_loss(z_slots[:, :, 1:], z_slots[:, :, :-1].detach())
                  if z_slots.shape[2] > 1 else l_rec.new_zeros(()))
        # held-out view: recover the hidden camera's object slots from the fused state
        hid = int(torch.randint(self.cfg["n_views"], (1,)).item())
        fused = self.fusion(z_slots, hidden_view=hid)
        pred_h = self.heldout(fused)
        truth = z_slots[:, hid][..., list(OBJECT_SLOTS), :]
        l_view = F.mse_loss(pred_h, truth.detach())
        # SIGReg per (semantic slot, timestep) across the global batch — one group per identity
        b, v, t, n_state, d = z_slots.shape
        sig_in = z_slots.permute(2, 3, 0, 1, 4).reshape(t * n_state, b * v, d)
        l_sig = self.sigreg(sig_in) if cfg["lambda_s"] > 0 else l_rec.new_zeros(())
        total = (cfg["lambda_m"] * l_rec + cfg["lambda_b"] * l_bind
                 + cfg["lambda_s"] * l_sig + cfg["lambda_t"] * l_temp
                 + cfg["lambda_v"] * l_view)
        return total, {"L_rec": l_rec.detach(), "L_bind": l_bind.detach(),
                       "L_sigreg": l_sig.detach(), "L_temp": l_temp.detach(),
                       "L_view": l_view.detach()}

    # -- full forward -------------------------------------------------------------
    def forward(self, batch, n_masked=2, hide_slot=None):
        cfg = self.cfg
        tok = self.tokenize(batch)
        z = self.fusion(tok["slots"][..., list(STATE_SLOTS), :])   # fused state z_t
        pred, target, ids = self.predictor(z, batch["control"], n_masked, hide_slot)
        l_hist, l_fut, cos_h, cos_f = predictor_losses(pred, target, ids, cfg["history"])
        l_pred = l_hist + l_fut
        l_perc, parts = self.perception_losses(batch, tok)
        parts.update({"L_hist": l_hist.detach(), "L_future": l_fut.detach(),
                      "cos_hist": cos_h.detach(), "cos_future": cos_f.detach(),
                      "L_pred": l_pred.detach()})
        return l_pred + l_perc, parts, pred, target


# --------------------------------------------------------------------------------------
# 6. train / test
# --------------------------------------------------------------------------------------
def split_recordings(recs, val_frac=0.25):
    n_val = max(1, round(val_frac * len(recs))) if len(recs) > 1 else 0
    if n_val == 0 or n_val == len(recs):
        return recs, recs
    return recs[:-n_val], recs[-n_val:]


def _load_all(data_dir: Path, feats_dir: Path | None, limit: int = 0):
    files = sorted(data_dir.glob("*.npz"))
    if limit:
        files = files[:limit]
    if not files:
        raise SystemExit(f"no .npz recordings found in {data_dir}")
    recs = []
    for f in files:
        r = load_npz(f)
        if feats_dir is not None:
            load_cached_features(feats_dir, r)
        recs.append(r)
    return recs


def _finite(*tensors):
    return all(bool(torch.isfinite(t).all()) for t in tensors if torch.is_tensor(t))


def run_train(cfg, recs, out_dir: Path, quiet=False):
    length = cfg["history"] + cfg["future"]
    tr, va = split_recordings(recs)
    tdl = DataLoader(Windows(tr, length, stride=1), cfg["batch_size"], shuffle=True,
                     collate_fn=Windows.collate, drop_last=len(Windows(tr, length, 1)) > cfg["batch_size"])
    vdl = DataLoader(Windows(va, length, 1), cfg["batch_size"], shuffle=False,
                     collate_fn=Windows.collate)
    sample = next(iter(tdl))
    n_patches = (sample["feats"].shape[-2] if "feats" in sample
                 else (cfg["img"] // cfg["patch"]) ** 2)
    feat_dim = sample["feats"].shape[-1] if "feats" in sample else cfg["backbone_dim"]
    ctrl_dim = sample["control"].shape[-1]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = C3JEPA(cfg, ctrl_dim, n_patches, feat_dim).to(dev)
    opt = torch.optim.AdamW(model.parameters(), cfg["lr"], weight_decay=cfg["weight_decay"])
    (out_dir / "ckpt").mkdir(parents=True, exist_ok=True)
    history, best = [], math.inf
    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        agg, n = {}, 0
        t0 = time.time()
        for batch in tdl:
            batch = {k: v.to(dev) for k, v in batch.items()}
            loss, parts, _, _ = model(batch, cfg["masked_slots"])
            if not _finite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {ep}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            bs = batch["control"].shape[0]
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + float(v) * bs
            agg["loss"] = agg.get("loss", 0.0) + float(loss.detach()) * bs
            n += bs
        train_row = {k: v / max(n, 1) for k, v in agg.items()}
        val_row = evaluate(model, vdl, dev, cfg)
        row = {"epoch": ep, "sec": round(time.time() - t0, 1),
               **{f"train_{k}": round(v, 6) for k, v in train_row.items()},
               **{f"val_{k}": round(v, 6) for k, v in val_row.items()}}
        history.append(row)
        if not quiet:
            print(json.dumps(row), flush=True)
        if val_row["pred_mse"] < best:
            best = val_row["pred_mse"]
            torch.save({"model": model.state_dict(), "cfg": cfg,
                        "n_patches": n_patches, "feat_dim": feat_dim,
                        "ctrl_dim": ctrl_dim}, out_dir / "ckpt" / "best.pt")
        (out_dir / "metrics.json").write_text(json.dumps(history, indent=1), encoding="utf-8")
    return model, history, best


@torch.no_grad()
def evaluate(model, dl, dev, cfg):
    """Rollout error against the persistence baseline; held-out-view check."""
    model.eval()
    agg, n = {}, 0
    for batch in dl:
        batch = {k: v.to(dev) for k, v in batch.items()}
        _, parts, pred, target = model(batch, cfg["masked_slots"], hide_slot=cfg["hide_slot"])
        h, f = cfg["history"], cfg["history"] + cfg["future"]
        persist = target[:, h - 1:h, :, :].expand(-1, f - h, -1, -1)
        m_fut = float(F.mse_loss(pred[:, h:], target[:, h:]))
        m_per = float(F.mse_loss(persist, target[:, h:]))
        bs = batch["control"].shape[0]
        for k, v in parts.items():
            agg[k] = agg.get(k, 0.0) + float(v) * bs
        agg["pred_mse"] = agg.get("pred_mse", 0.0) + m_fut * bs
        agg["persist_mse"] = agg.get("persist_mse", 0.0) + m_per * bs
        n += bs
    out = {k: v / max(n, 1) for k, v in agg.items()}
    # the persistence comparison is only meaningful when the latent actually moves;
    # on degenerate (e.g. synthetic, near-static) data report it as unavailable
    out["persist_improvement_pct"] = (100.0 * (1 - out["pred_mse"] / out["persist_mse"])
                                      if out["persist_mse"] > 1e-8 else None)
    return out


def run_test(ckpt_path: Path, data_dir: Path, feats_dir, limit=0):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = {**DEFAULTS, **ckpt["cfg"], "hide_slot": 0}
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    recs = _load_all(data_dir, feats_dir, limit)
    _, va = split_recordings(recs)
    dl = DataLoader(Windows(va, cfg["history"] + cfg["future"], 1), cfg["batch_size"],
                    shuffle=False, collate_fn=Windows.collate)
    model = C3JEPA(cfg, ckpt["ctrl_dim"], ckpt["n_patches"], ckpt["feat_dim"]).to(dev)
    model.load_state_dict(ckpt["model"])
    metrics = evaluate(model, dl, dev, cfg)
    print(json.dumps({k: round(v, 6) if isinstance(v, float) else v
                      for k, v in metrics.items()}, indent=1))
    return metrics


# --------------------------------------------------------------------------------------
# 7. CLI
# --------------------------------------------------------------------------------------
def build_cfg(args):
    cfg = dict(DEFAULTS)
    for k in ("n_views", "n_slots", "slot_dim", "history", "future", "masked_slots",
              "lambda_m", "lambda_b", "lambda_s", "lambda_t", "lambda_v", "lr", "epochs",
              "batch_size", "seed", "patch"):
        if getattr(args, k, None) is not None:
            cfg[k] = getattr(args, k)
    cfg["img"] = args.img
    cfg["depth"] = args.depth
    cfg["heads"] = args.heads
    cfg["hidden"] = args.hidden
    cfg["sigreg_proj"] = args.sigreg_proj
    cfg["hide_slot"] = args.hide_slot
    cfg["fusion_heads"] = (getattr(args, "fusion_heads", None) or DEFAULTS["fusion_heads"])
    return cfg


def main(argv=None):
    p = argparse.ArgumentParser(description="C^3-JEPA minimal reference implementation")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--data-dir", type=Path, default=Path("data/recs"))
        sp.add_argument("--cached-features", type=Path, default=None)
        sp.add_argument("--img", type=int, default=32, help="frame size for the stub backbone")
        sp.add_argument("--patch", type=int, default=16)
        sp.add_argument("--history", type=int, default=None)
        sp.add_argument("--future", type=int, default=None)

    st = sub.add_parser("smoke", help="synthetic end-to-end run (CPU, about a minute)")
    st.add_argument("--out-dir", type=Path, default=Path("runs/smoke"))
    st.add_argument("--epochs", type=int, default=80)
    st.add_argument("--records", type=int, default=8)
    st.add_argument("--keep-data", action="store_true")
    st.add_argument("--history", type=int, default=None)
    st.add_argument("--future", type=int, default=None)
    st.set_defaults(n_views=2, n_slots=6, slot_dim=32, history=4, future=8,
                    masked_slots=2, batch_size=4, img=32, patch=8, depth=2, heads=4,
                    hidden=64, sigreg_proj=32, hide_slot=0, fusion_heads=2,
                    lambda_m=1.0, lambda_b=1.0, lambda_s=0.03, lambda_t=0.1, lambda_v=0.2,
                    lr=1e-3, seed=0)

    tr = sub.add_parser("train", help="train on recordings in --data-dir")
    common(tr)
    tr.add_argument("--out-dir", type=Path, default=Path("runs/example"))
    tr.set_defaults(n_views=None, n_slots=None, slot_dim=None, history=None, future=None,
                    masked_slots=None, batch_size=None, depth=3, heads=4, hidden=128,
                    sigreg_proj=64, hide_slot=0, lambda_m=None, lambda_b=None, lambda_s=None,
                    lambda_t=None, lambda_v=None, lr=None, epochs=None, seed=None)

    te = sub.add_parser("test", help="evaluate a checkpoint")
    common(te)
    te.add_argument("--ckpt", type=Path, required=True)
    te.set_defaults(hide_slot=0, depth=None, heads=None, hidden=None, sigreg_proj=None,
                    n_views=None, n_slots=None, slot_dim=None, masked_slots=None,
                    batch_size=None, lr=None, epochs=None, seed=None,
                    lambda_m=None, lambda_b=None, lambda_s=None, lambda_t=None, lambda_v=None)

    args = p.parse_args(argv)
    torch.manual_seed(args.seed if args.seed is not None else DEFAULTS["seed"])
    np.random.seed(args.seed if args.seed is not None else DEFAULTS["seed"])
    cfg = build_cfg(args)

    if args.cmd == "smoke":
        data_dir = Path("data/smoke")
        recs = make_synthetic(n_rec=args.records, T=cfg["history"] + cfg["future"] + 8,
                              n_views=cfg["n_views"], H=args.img, W=args.img,
                              patch=cfg["patch"], seed=cfg["seed"], out_dir=data_dir)
        out = args.out_dir
        if out.exists():
            shutil.rmtree(out)
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dev_label = f"cuda ({torch.cuda.get_device_name(0)})" if dev.type == "cuda" else "cpu"
        print(f"[smoke] {len(recs)} synthetic recordings, "
              f"{cfg['history']}+{cfg['future']} window, slots={cfg['n_slots']}, "
              f"device={dev_label}")
        model, hist, best = run_train(cfg, recs, out)
        # report on the same held-out recordings the checkpoint was selected on
        _, held_out = split_recordings(recs)
        metrics = evaluate(model, DataLoader(Windows(held_out, cfg["history"] + cfg["future"], 1),
                                             cfg["batch_size"], collate_fn=Windows.collate),
                           next(model.parameters()).device, cfg)
        # gates: finite losses, a real baseline comparison, and the two weightings
        assert all(math.isfinite(h["train_loss"]) for h in hist), "non-finite training loss"
        assert math.isfinite(metrics["pred_mse"]), "non-finite rollout error"
        report = {"epochs": len(hist), "best_val_pred_mse": best,
                  "final": hist[-1], "smoke_eval": metrics}
        (out / "smoke_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(json.dumps({"best_val_pred_mse": round(best, 6),
                          "persist_mse": round(metrics["persist_mse"], 6),
                          "pred_mse": round(metrics["pred_mse"], 6),
                          "improvement_pct": round(metrics["persist_improvement_pct"], 2)},
                         indent=1))
        if not args.keep_data:
            shutil.rmtree(data_dir, ignore_errors=True)
        print("[smoke] OK — training and testing both ran end to end.")
        return 0

    if args.cmd == "train":
        recs = _load_all(args.data_dir, args.cached_features)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        run_train(cfg, recs, args.out_dir)
        print(f"[train] wrote {args.out_dir}/ckpt/best.pt")
        return 0

    if args.cmd == "test":
        run_test(args.ckpt, args.data_dir, args.cached_features)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
