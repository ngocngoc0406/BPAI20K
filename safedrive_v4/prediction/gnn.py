"""Direction B: temporal relational GNN (R-GCN per snapshot -> GRU over the window), trained on the SAME folds, samples and
evaluation windows as the LightGBM configurations (v3 protocol). Fixed hyper-parameters and epochs (no tuning on evaluation folds).

Model: object encoder MLP(21->32), ego encoder MLP(5->32); 2 relational graph-convolution layers over the 11 typed, confidence-
weighted scene-graph relations (both directions); readout = [ego node, masked mean of objects]; GRU(64) over the W = 1 s window
(10 steps); linear -> logit. ~40k parameters.
Usage: python -m safedrive_v4.prediction.gnn [--epochs 6]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from safedrive_v4.prediction.graph_tensors import F_EGO, F_OBJ, N_OBJ, RELS, tensor_dir  # noqa: E402

N_STEPS = 10


class TemporalRGCN(nn.Module):
    def __init__(self, hid: int = 32, n_rel: int = len(RELS)):
        super().__init__()
        self.obj = nn.Sequential(nn.Linear(F_OBJ, hid), nn.ReLU(), nn.Linear(hid, hid))
        self.ego = nn.Sequential(nn.Linear(F_EGO, hid), nn.ReLU(), nn.Linear(hid, hid))
        self.layers = nn.ModuleList()
        for _ in range(2):
            self.layers.append(nn.ModuleDict({"self": nn.Linear(hid, hid), "rel": nn.Linear(hid * 2 * n_rel, hid)}))
        self.gru = nn.GRU(2 * hid, 64, batch_first=True)
        self.out = nn.Linear(64, 1)

    def forward(self, ego, x, mask, adj):
        # ego (B,T,5) x (B,T,N,21) mask (B,T,N) adj (B,T,R,N+1,N+1)
        B, T = ego.shape[:2]
        h = torch.cat([self.ego(ego).unsqueeze(2), self.obj(x) * mask.unsqueeze(-1)], 2)      # (B,T,N+1,H)
        A = adj
        At = A.transpose(-1, -2)
        An = A / (A.sum(-1, keepdim=True) + 1e-6)          # out-degree normalised (messages along edges)
        Atn = At / (At.sum(-1, keepdim=True) + 1e-6)       # in-degree normalised (messages against edges)
        for L in self.layers:
            fwd = torch.einsum("btrij,btjh->btrih", An, h)
            bwd = torch.einsum("btrij,btjh->btrih", Atn, h)
            m = torch.cat([fwd, bwd], 2).permute(0, 1, 3, 2, 4).reshape(B, T, h.shape[2], -1)
            h = torch.relu(L["self"](h) + L["rel"](m))
        full = torch.cat([torch.ones_like(mask[..., :1]), mask], 2).unsqueeze(-1)
        pooled = (h[:, :, 1:] * mask.unsqueeze(-1)).sum(2) / mask.sum(2, keepdim=True).clamp_min(1)
        z = torch.cat([h[:, :, 0], pooled], -1) * full[:, :, 0]
        o, _ = self.gru(z)
        return self.out(o[:, -1]).squeeze(-1)


class Store:
    def __init__(self, vids):
        d = tensor_dir()
        self.t = {}
        for v in vids:
            with np.load(d / f"{v}.npz") as z:
                self.t[v] = (z["ego"].astype(np.float32), z["x"].astype(np.float32), z["mask"].astype(np.float32), z["adj"].astype(np.float32))

    def windows(self, vid, times, t_ends):
        from src.prediction.windows import window_indices
        idx = np.stack([window_indices(times, float(t), N_STEPS) for t in t_ends])
        e, x, m, a = self.t[vid]
        return e[idx], x[idx], m[idx], a[idx]


def batches(store, items, times_of, bs=256):
    """items: list of (vid, t_end). Yields tensors grouped by video for efficiency."""
    by = {}
    for i, (v, t) in enumerate(items):
        by.setdefault(v, []).append((i, t))
    order, E, X, M, A = [], [], [], [], []
    for v, lst in by.items():
        e, x, m, a = store.windows(v, times_of[v], [t for _, t in lst])
        order += [i for i, _ in lst]
        E.append(e), X.append(x), M.append(m), A.append(a)
    E, X, M, A = (np.concatenate(z) for z in (E, X, M, A))
    return np.array(order), E, X, M, A


def fit_predict(train_items, y, pred_sets, store, times_of, epochs=6, seed=42, device="cuda"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    net = TemporalRGCN().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    order, E, X, M, A = batches(store, train_items, times_of)
    yy = torch.tensor(y[order], dtype=torch.float32)
    pw = torch.tensor([(yy == 0).sum() / max((yy == 1).sum(), 1)], device=dev)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    T = [torch.tensor(z) for z in (E, X, M, A)]
    n = len(yy)
    for ep in range(epochs):
        perm = torch.randperm(n)
        net.train()
        for i in range(0, n, 256):
            b = perm[i:i + 256]
            opt.zero_grad()
            loss = lossf(net(*(t[b].to(dev) for t in T)), yy[b].to(dev))
            loss.backward()
            opt.step()
    del T
    net.eval()
    outs = []
    with torch.no_grad():
        for items in pred_sets:
            p = np.zeros(len(items), np.float32)
            by: dict[str, list[int]] = {}
            for i, (v, _) in enumerate(items):
                by.setdefault(v, []).append(i)
            for v, ii in by.items():   # per video, small batches: bounded GPU and CPU memory (4 GB GPU)
                z = store.windows(v, times_of[v], [items[i][1] for i in ii])
                res = [torch.sigmoid(net(*(torch.tensor(q[j:j + 256]).to(dev) for q in z))).cpu().numpy() for j in range(0, len(ii), 256)]
                p[ii] = np.concatenate(res)
            outs.append(p)
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return outs, sum(p.numel() for p in net.parameters())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    a = ap.parse_args()
    from run_ew_study import Study
    from safedrive_v4.evaluation.ablation import LABELS, V3_GROUPS, V4Study, window_metrics
    from src.evaluation.metrics import classification_metrics, nexar_map
    from src.prediction.windows import train_samples
    st = V4Study(False)
    store = Store(list(st.feats))
    times_of = {v: f.times for v, f in st.feats.items()}
    cfg = st.cfg_for(LABELS)
    y_all, g_all, p_all, streams = [], [], [], {}
    t0 = time.perf_counter()
    n_params = None
    for k in range(5):
        tr_recs = [st.recs[v] for v in st.dev if st.fold[v] != k]
        r2 = np.random.default_rng(600 + k)
        tr = sum((train_samples(tr_recs, st.feats, cfg, r2) for _ in range(3)), [])
        ev = st.oof_ev[k]
        sv = [v for v in st.dev if st.fold[v] == k]
        stream_items = [(v, float(t)) for v in sv for t in st.feats[v].times[np.arange(0, len(st.feats[v].times), st.stride)]]
        (pe, ps), n_params = fit_predict([(s.video_id, s.t_end) for s in tr], np.array([s.label for s in tr]),
                                         [[(s.video_id, s.t_end) for s in ev], stream_items], store, times_of, a.epochs, 42 + k)
        y_all.append([s.label for s in ev]), g_all.append([s.group for s in ev]), p_all.append(pe)
        j = 0
        for v in sv:
            n = len(np.arange(0, len(st.feats[v].times), st.stride))
            streams[v] = {"times": st.feats[v].times[np.arange(0, len(st.feats[v].times), st.stride)], "score": ps[j:j + n]}
            j += n
        print(f"fold {k} done {time.perf_counter() - t0:.0f}s", flush=True)
    y, g, p = np.concatenate(y_all).astype(float), np.concatenate(g_all), np.concatenate(p_all)
    gnn = {"y": y, "g": g, "p": p, "streams": streams, "oof_mAP": nexar_map(y, p, g, st.cfg["windows"]["eval_horizons_s"])["mAP"],
           "roc_auc": classification_metrics(y, p, 0.5)["roc_auc"]}
    ref = st.run(V3_GROUPS, 1.0, LABELS, "lgbm")
    sc = st.score(gnn, 1.0)
    res = {"model": "TemporalRGCN", "params": int(n_params), "epochs": a.epochs, "train_minutes": round((time.perf_counter() - t0) / 60, 1),
           "oof_raw": window_metrics(y, p, g, st.cfg["windows"]["eval_horizons_s"]), "S": sc["S"],
           **{kk: sc[kk] for kk in sc if kk.startswith(("recall@", "edr@"))},
           "dS_vs_A0": st.paired(gnn, ref, 1.0, 1.0), "dmAP_vs_A0": st.paired_map(gnn, ref)}
    (ROOT / "experiments/v4/gnn_oof.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(json.dumps(res, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
