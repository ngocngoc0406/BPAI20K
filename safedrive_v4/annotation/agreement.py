"""Inter-annotator agreement for ann-v4.0, with a metric per label type (docs/SAFEDRIVE_V4_ANNOTATION_DESIGN.md §5).

  events      greedy one-to-one matching by temporal IoU >= 0.3 (same algorithm as v3 `annotation.match`)
              -> Cohen's kappa on categories (matched; and with unmatched events as 'none'), kappa event vs not,
                 event-detection F1 between annotators, mean tIoU, boundary deviations (start / peak / end, seconds)
  actors      matched per event pair by tracker id, else box IoU >= 0.3 at a common reference time -> actor-set F1
  relations   (subject, object, relation) tuples with mapped actors and tIoU >= 0.3 -> relation F1; kappa of the relation
              type only for matched actor pairs whose intervals overlap
  ordinal     quadratic-weighted kappa for visibility / occlusion / blur / lighting / confidence on matched events
  uncertainty share of 'uncertain' events and of low-confidence labels per annotator
Kappa is NOT used for boundaries, sets or intervals. Disagreements are listed, never resolved automatically.
"""
from __future__ import annotations

import numpy as np

from safedrive_v4.annotation.schema import EVENT_CATEGORIES, ORDINAL
from src.realworld.annotation import cohen_kappa  # v3, read-only


def tiou(a0, a1, b0, b1) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else float(a0 == b0)


def match_intervals(A: list, B: list, start="t_start", end="t_end", min_tiou: float = 0.3):
    cand = sorted(((tiou(getattr(a, start), getattr(a, end), getattr(b, start), getattr(b, end)), i, j)
                   for i, a in enumerate(A) for j, b in enumerate(B)), reverse=True)
    ua, ub, pairs = set(range(len(A))), set(range(len(B))), []
    for t, i, j in cand:
        if t < min_tiou:
            break
        if i in ua and j in ub:
            ua.discard(i), ub.discard(j)
            pairs.append((A[i], B[j], t))
    return pairs, [A[i] for i in sorted(ua)], [B[j] for j in sorted(ub)]


def weighted_kappa(x: list[str], y: list[str], levels: tuple) -> float:
    """Quadratic-weighted Cohen's kappa for ordinal labels."""
    if not x:
        return float("nan")
    k = len(levels)
    idx = {c: i for i, c in enumerate(levels)}
    O = np.zeros((k, k))
    for a, b in zip(x, y):
        O[idx[a], idx[b]] += 1
    W = np.array([[(i - j) ** 2 / (k - 1) ** 2 for j in range(k)] for i in range(k)])
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    den = (W * E).sum()
    return float(1 - (W * O).sum() / den) if den > 0 else 1.0


def _box_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def map_actors(actA: list, actB: list) -> dict[str, str]:
    """label in A -> label in B (ego <-> ego; tracker id equality; else box IoU >= 0.3 at reference times within 0.5 s)."""
    out = {}
    usedB = set()
    for a in actA:
        best, bi = None, 0.0
        for b in actB:
            if b.label in usedB:
                continue
            if a.cls == "ego" and b.cls == "ego":
                best, bi = b, 1.0
                break
            if a.track_id is not None and a.track_id == b.track_id:
                best, bi = b, 1.0
                break
            if a.box and b.box and a.ref_frame_t is not None and b.ref_frame_t is not None and abs(a.ref_frame_t - b.ref_frame_t) <= 0.5:
                v = _box_iou(a.box, b.box)
                if v >= 0.3 and v > bi:
                    best, bi = b, v
        if best is not None:
            out[a.label] = best.label
            usedB.add(best.label)
    out.setdefault("ego", "ego")
    return out


def f1(tp: int, fp: int, fn: int) -> float:
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")


def video_agreement(store, video: str, a: str, b: str, min_tiou: float = 0.3) -> dict:
    EA, EB = store.records(video, a, "event"), store.records(video, b, "event")
    pairs, ua, ub = match_intervals(EA, EB, min_tiou=min_tiou)
    xa, xb = [p[0].category for p in pairs], [p[1].category for p in pairs]
    xa_u = xa + [e.category for e in ua] + ["none"] * len(ub)
    xb_u = xb + ["none"] * len(ua) + [e.category for e in ub]
    evf = lambda c: "event" if c in EVENT_CATEGORIES else "no_event"  # noqa: E731
    dev = {k: [abs(getattr(p[0], k) - getattr(p[1], k)) for p in pairs if getattr(p[0], k) is not None and getattr(p[1], k) is not None]
           for k in ("t_start", "t_peak", "t_end")}
    # actors and relations
    amap = map_actors(store.records(video, a, "actor"), store.records(video, b, "actor"))
    actor_tp = actor_fp = actor_fn = 0
    for ea, eb, _ in pairs:
        mapped = {amap.get(x) for x in ea.actors}
        sb = set(eb.actors)
        tp = len(mapped & sb)
        actor_tp, actor_fp, actor_fn = actor_tp + tp, actor_fp + len(ea.actors) - tp, actor_fn + len(sb) - tp
    IA, IB = store.records(video, a, "interaction"), store.records(video, b, "interaction")
    rel_tp, used, rel_types = 0, set(), ([], [])
    for ia in IA:
        sub, obj = amap.get(ia.subject), (amap.get(ia.object) if ia.object not in (None, "ego") else ia.object)
        best = None
        for j, ib in enumerate(IB):
            if j in used or ib.subject != sub or ib.object != obj:
                continue
            t = tiou(ia.t_start, ia.t_end, ib.t_start, ib.t_end)
            if t > 0 and (best is None or t > best[1]):
                best = (j, t, ib)
        if best is not None:
            rel_types[0].append(ia.relation), rel_types[1].append(best[2].relation)
            if best[2].relation == ia.relation and best[1] >= min_tiou:
                rel_tp += 1
                used.add(best[0])
    ordinal = {k: weighted_kappa([p[0].uncertainty[k] for p in pairs], [p[1].uncertainty[k] for p in pairs], lv)
               for k, lv in ORDINAL.items()} if pairs else {k: float("nan") for k in ORDINAL}
    disagreements = ([{"type": "only_" + a, "event": e.record_id, "category": e.category, "t": [e.t_start, e.t_end]} for e in ua] +
                     [{"type": "only_" + b, "event": e.record_id, "category": e.category, "t": [e.t_start, e.t_end]} for e in ub] +
                     [{"type": "category", "events": [p[0].record_id, p[1].record_id], "categories": [p[0].category, p[1].category]}
                      for p in pairs if p[0].category != p[1].category] +
                     [{"type": "boundary > 1 s", "events": [p[0].record_id, p[1].record_id]}
                      for p in pairs if max(abs(p[0].t_start - p[1].t_start), abs(p[0].t_end - p[1].t_end)) > 1.0])

    def unc(E):
        return {"uncertain_share": float(np.mean([e.category == "uncertain" for e in E])) if E else float("nan"),
                "low_confidence_share": float(np.mean([e.uncertainty["annotation_confidence"] == "low" for e in E])) if E else float("nan")}
    stat = lambda d: {"n": len(d), "mean_abs_s": float(np.mean(d)) if d else None, "median_abs_s": float(np.median(d)) if d else None}  # noqa: E731
    return {"video": video, "annotators": [a, b], "events": [len(EA), len(EB)], "matched": len(pairs),
            "event_detection_f1": f1(len(pairs), len(ua), len(ub)),
            "kappa_category_matched": cohen_kappa(xa, xb), "kappa_category_with_unmatched": cohen_kappa(xa_u, xb_u),
            "kappa_event_vs_not": cohen_kappa([evf(c) for c in xa_u], [evf(c) for c in xb_u]),
            "mean_tiou": float(np.mean([p[2] for p in pairs])) if pairs else float("nan"),
            "boundary_deviation": {k: stat(v) for k, v in dev.items()},
            "actor_f1": f1(actor_tp, actor_fp, actor_fn), "relation_f1": f1(rel_tp, len(IA) - rel_tp, len(IB) - rel_tp),
            "kappa_relation_type_matched_pairs": cohen_kappa(*rel_types) if rel_types[0] else float("nan"),
            "ordinal_weighted_kappa": ordinal, "uncertainty": {a: unc(EA), b: unc(EB)}, "disagreements": disagreements}


def dataset_agreement(store, a: str, b: str) -> dict:
    """Pooled agreement over every video both annotators SUBMITTED (pooled labels, not an average of per-video kappas)."""
    vids = [v for v in store.videos() if store.submitted(v, a) and store.submitted(v, b)]
    per = [video_agreement(store, v, a, b) for v in vids]
    if not per:
        return {"videos": 0, "status": "NOT EVALUATED: no video annotated and submitted by both annotators"}
    xa, xb = [], []
    for v in vids:
        pairs, ua, ub = match_intervals(store.records(v, a, "event"), store.records(v, b, "event"))
        xa += [p[0].category for p in pairs] + [e.category for e in ua] + ["none"] * len(ub)
        xb += [p[1].category for p in pairs] + ["none"] * len(ua) + [e.category for e in ub]
    mean = lambda k: float(np.nanmean([p[k] for p in per]))  # noqa: E731
    return {"videos": len(vids), "kappa_category_pooled_with_unmatched": cohen_kappa(xa, xb),
            "event_detection_f1_mean": mean("event_detection_f1"), "mean_tiou": mean("mean_tiou"),
            "actor_f1_mean": mean("actor_f1"), "relation_f1_mean": mean("relation_f1"),
            "disagreements": sum(len(p["disagreements"]) for p in per), "per_video": per}
