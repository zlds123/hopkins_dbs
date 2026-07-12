"""Two-tower InfoNCE alignment for AJILE12 ECoG (neural tower) and pose (behavior tower).

Phase 3 core model (M3): a symmetric contrastive encoder pair, CLIP-style, that maps
time-matched neural band-power and wrist pose/kinematics into a shared cosine space.
Positives are (by default) exact same-time (neural, behavior) pairs; negatives are the
rest of the minibatch (in-batch negatives -- no memory bank, appropriate for the data
scale and CPU-only budget here). Unlike CEBRA (single input stream + auxiliary), both
towers here are full encoders, so the shared space is queryable from either side --
needed for the bidirectional-decode question (H3.1).

Mirrors the fit/transform/cache API of ``cebra_ajile.py``'s ``fit_cebra`` and
``phase1_resolution.py``'s ``build_cebra_matrix`` so ``phase3_eval.py`` can call
M1 (CEBRA-Time), M2 (CEBRA-Behavior), and M3 (two-tower) uniformly.

Fitting convention: trains on ``train_idx`` only (never the held-out test split), since
the behavior tower's inputs (position/velocity/speed) are also Phase 3 decode targets
(T2-T4) -- fitting on the full stream would leak test-fold pose structure into the
self-supervised representation before the label-efficiency / blocked-decode numbers are
even computed.
"""

import hashlib
import os

import numpy as np


def build_behavior_matrix(stream):
    """(T, 5K) = concat(pos, vel, speed) for the K pose keypoints in the stream."""
    parts = [stream["pos"], stream["vel"], stream["speed"]]
    parts = [np.asarray(p, dtype=np.float32) for p in parts if p is not None and p.shape[1]]
    if not parts:
        raise ValueError("stream has no pose data (pos/vel/speed all empty)")
    return np.concatenate(parts, axis=1)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def _mlp(in_dim, hidden, out_dim, torch, nn):
    layers = []
    d = in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.ReLU()]
        d = h
    layers += [nn.Linear(d, out_dim)]
    return nn.Sequential(*layers)


def _make_towers(neural_dim, behavior_dim, dim, torch, nn):
    class Tower(nn.Module):
        def __init__(self, in_dim, hidden):
            super().__init__()
            self.net = _mlp(in_dim, hidden, dim, torch, nn)

        def forward(self, x):
            z = self.net(x)
            return nn.functional.normalize(z, dim=-1)

    class TwoTowerModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.neural = Tower(neural_dim, (128, 128))
            self.behavior = Tower(behavior_dim, (64, 64))

        def encode_neural(self, x):
            return self.neural(x)

        def encode_behavior(self, b):
            return self.behavior(b)

    return TwoTowerModel()


def info_nce_loss(z_n, z_b, temperature, torch, nn):
    """Symmetric (CLIP-style) InfoNCE over an in-batch similarity matrix."""
    logits = z_n @ z_b.T / temperature
    target = torch.arange(logits.shape[0], device=logits.device)
    loss_n = nn.functional.cross_entropy(logits, target)
    loss_b = nn.functional.cross_entropy(logits.T, target)
    return 0.5 * (loss_n + loss_b)


# --------------------------------------------------------------------------- #
# Fit / transform
# --------------------------------------------------------------------------- #
def fit_two_tower(X, B, dim, train_idx, time_offset=0, temperature=0.1,
                   batch_size=512, max_iter=2000, lr=3e-4, weight_decay=1e-5,
                   seed=0, device="cpu", verbose=True):
    """Train a two-tower InfoNCE model on ``train_idx`` only; transform the full stream.

    Parameters
    ----------
    X : (T, C) neural feature stream (already standardized upstream)
    B : (T, D) behavior feature stream (from ``build_behavior_matrix``)
    dim : output embedding dimension (both towers)
    train_idx : indices used for fitting (e.g. the first 70% of time)
    time_offset : if > 0, positive behavior index for anchor t is sampled uniformly
        from [t-time_offset, t+time_offset] instead of exactly t.

    Returns
    -------
    model, Z_n (T, dim), Z_b (T, dim)
    """
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    X = np.asarray(X, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)
    train_idx = np.asarray(train_idx, dtype=np.int64)
    n_train = len(train_idx)
    bs = min(batch_size, max(2, n_train))

    model = _make_towers(X.shape[1], B.shape[1], dim, torch, nn).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    Xt = torch.from_numpy(X[train_idx]).to(device)
    Bt = torch.from_numpy(B[train_idx]).to(device)
    lo_bound, hi_bound = 0, n_train - 1

    model.train()
    for it in range(max_iter):
        anchor = rng.choice(n_train, size=bs, replace=False)
        if time_offset > 0:
            jitter = rng.integers(-time_offset, time_offset + 1, size=bs)
            beh_idx = np.clip(anchor + jitter, lo_bound, hi_bound)
        else:
            beh_idx = anchor

        xb = Xt[anchor]
        bb = Bt[beh_idx]
        z_n = model.encode_neural(xb)
        z_b = model.encode_behavior(bb)
        loss = info_nce_loss(z_n, z_b, temperature, torch, nn)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if verbose and (it + 1) % max(1, max_iter // 10) == 0:
            print("  two-tower iter {}/{}  loss={:.4f}".format(it + 1, max_iter, float(loss.detach())))

    model.eval()
    with torch.no_grad():
        Z_n = model.encode_neural(torch.from_numpy(X).to(device)).cpu().numpy()
        Z_b = model.encode_behavior(torch.from_numpy(B).to(device)).cpu().numpy()
    return model, Z_n.astype(np.float64), Z_b.astype(np.float64)


def transform(model, X=None, B=None, device="cpu"):
    """Frozen inference with an already-fit model. Either input may be None."""
    import torch

    model.eval()
    z_n = z_b = None
    with torch.no_grad():
        if X is not None:
            z_n = model.encode_neural(
                torch.from_numpy(np.asarray(X, dtype=np.float32)).to(device)).cpu().numpy()
        if B is not None:
            z_b = model.encode_behavior(
                torch.from_numpy(np.asarray(B, dtype=np.float32)).to(device)).cpu().numpy()
    return z_n, z_b


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #
def get_two_tower(X, B, cache_dir, dim, train_idx, time_offset=0, temperature=0.1,
                   batch_size=512, max_iter=2000, lr=3e-4, seed=0, tag="",
                   verbose=True):
    """Fit-or-load, keyed by hyperparameters + data shape (md5-hash, mirrors
    ``phase1_resolution.get_stream``'s caching pattern)."""
    import torch

    os.makedirs(cache_dir, exist_ok=True)
    key = "{}|{}|{}|{}|{}|{:.4g}|{:.4g}|{}|{:.2e}|{}|{}".format(
        X.shape, B.shape, dim, len(train_idx), time_offset, temperature,
        batch_size, max_iter, lr, seed, tag)
    h = hashlib.md5(key.encode()).hexdigest()[:10]
    npz_path = os.path.join(cache_dir, "two_tower_{}.npz".format(h))
    pt_path = os.path.join(cache_dir, "two_tower_{}.pt".format(h))

    if os.path.exists(npz_path) and os.path.exists(pt_path):
        if verbose:
            print("loading cached two-tower:", npz_path)
        import torch.nn as nn
        d = np.load(npz_path, allow_pickle=True)
        model = _make_towers(X.shape[1], B.shape[1], dim, torch, nn)
        model.load_state_dict(torch.load(pt_path, map_location="cpu"))
        model.eval()
        return {"model": model, "z_n": d["z_n"], "z_b": d["z_b"], "dim": dim}

    model, Z_n, Z_b = fit_two_tower(
        X, B, dim, train_idx, time_offset=time_offset, temperature=temperature,
        batch_size=batch_size, max_iter=max_iter, lr=lr, seed=seed, verbose=verbose)
    np.savez_compressed(npz_path, z_n=Z_n, z_b=Z_b)
    torch.save(model.state_dict(), pt_path)
    if verbose:
        print("cached two-tower ->", npz_path)
    return {"model": model, "z_n": Z_n, "z_b": Z_b, "dim": dim}
