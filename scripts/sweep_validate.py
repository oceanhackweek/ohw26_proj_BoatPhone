#!/usr/bin/env python3
"""Is the tuned configuration real, or did picking the best of thousands find noise?

Tuning ~8 filter parameters against 26 scenes carrying 42 vessels is enough freedom
to fit noise, so the accuracy of the winning config is not by itself evidence.
CLAUDE.md invariant 4: a result that works is exactly what a selection artefact also
looks like. Two checks, both of which re-run the WHOLE selection procedure rather
than re-scoring one config:

  NULL   shuffle the truth counts across scenes and select again. Whatever accuracy
         the search reaches on shuffled labels is the floor that means nothing. The
         real number has to beat it.
  SPLIT  select on a random half of the scenes, score on the held-out half. The gap
         between the two is the overfit.

Emits a matrix of per-scene counts once and does all the resampling on it, so the
cost is the grid, not the grid times the resamples.
"""
import argparse, itertools, json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from sweep_score import (load_cache, load_truth, select, fuse_counts,
                         apply_persistence)


def build_matrix(cache_dir, geom, grid):
    """(configs, counts[n_config, n_scene], scene_ids, truth[n_scene])."""
    truth_map = load_truth()
    caches = {a: load_cache(cache_dir, a, geom) for a in ("toa", "sr")}
    sids = [s for s in sorted(caches["toa"]) if s[:13] in truth_map]
    truth = np.array([truth_map[s[:13]] for s in sids], float)

    configs, rows = [], []
    for conf, conf2, mask, mnl, mxl, mxa, npct, shm in itertools.product(*grid[:8]):
        if mnl >= mxl:
            continue
        keeps = {a: {s: select(r, conf if a == "toa" else conf2, mask,
                               mnl, mxl, mxa, npct, shm)
                     for s, r in c.items()} for a, c in caches.items()}
        for pers in grid[8]:
            k2 = {a: apply_persistence(caches[a], k, pers) for a, k in keeps.items()}
            for mode in grid[9]:
                cnt = (fuse_counts(caches, k2, mode) if mode in ("both", "any")
                       else {s: int(k.sum()) for s, k in k2[mode].items()})
                rows.append([cnt.get(s, 0) for s in sids])
                configs.append({"asset": mode, "conf_toa": conf, "conf_sr": conf2,
                                "mask": mask, "min_len": mnl, "max_len": mxl,
                                "max_aspect": mxa, "nir_pct": npct, "shore_m": shm,
                                "persist": pers, "geom": geom})
    return configs, np.array(rows, float), sids, truth


def err_acc(pred, true):
    """1 - sum|pred-true| / sum(true), floored at 0. Cannot cancel across scenes."""
    d = np.abs(pred - true)
    s = d.sum(axis=-1) if pred.ndim > 1 else d.sum()
    return np.maximum(0.0, 1 - s / max(true.sum(), 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--geom", default="s640_i640_bgr")
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    grid = [
        (0.35, 0.4, 0.45, 0.5, 0.55, 0.6),            # conf toa
        (0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25),   # conf sr
        ("none", "water"),                             # mask
        (0.0,), (30.0, 40.0, 50.0, 60.0), (99.0,),     # min_len, max_len, max_aspect
        (0.0, 0.9, 0.99), (0.0, 15.0, 30.0),           # nir_pct, shore_m
        (0, 2), ("both", "any", "toa", "sr"),          # persist, mode
    ]
    configs, M, sids, truth = build_matrix(args.cache_dir, args.geom, grid)
    print(f"{len(configs)} configs x {len(sids)} scenes; truth total {truth.sum():.0f}")

    full = err_acc(M, truth)
    best = int(np.argmax(full))
    print(f"\nBEST ON ALL SCENES  err_acc {full[best]:.3f}  "
          f"MAE {np.abs(M[best]-truth).mean():.2f}  pred {M[best].sum():.0f}")
    print(f"  {json.dumps(configs[best])}")

    rng = np.random.default_rng(args.seed)

    # NULL -- select on shuffled truth. This is the score the search reaches on
    # labels that carry no information; anything at or below it is not a result.
    null = []
    for _ in range(args.trials):
        sh = rng.permutation(truth)
        null.append(err_acc(M, sh).max())
    null = np.array(null)
    print(f"\nNULL (truth shuffled, {args.trials} trials)")
    print(f"  best-of-grid err_acc: mean {null.mean():.3f}  p95 {np.percentile(null,95):.3f}"
          f"  max {null.max():.3f}")
    print(f"  real {full[best]:.3f} beats {(full[best] > null).mean()*100:.0f}% of nulls")

    # SPLIT -- select on half, score on the other half.
    n = len(sids)
    tr_s, te_s, gaps = [], [], []
    for _ in range(args.trials):
        idx = rng.permutation(n)
        a, b = idx[:n // 2], idx[n // 2:]
        sel = int(np.argmax(err_acc(M[:, a], truth[a])))
        tr = err_acc(M[sel, a], truth[a]); te = err_acc(M[sel, b], truth[b])
        tr_s.append(tr); te_s.append(te); gaps.append(tr - te)
    print(f"\nSPLIT-HALF ({args.trials} random splits)")
    print(f"  selected-half err_acc  mean {np.mean(tr_s):.3f}")
    print(f"  HELD-OUT half err_acc  mean {np.mean(te_s):.3f}  "
          f"sd {np.std(te_s):.3f}  p05 {np.percentile(te_s,5):.3f}")
    print(f"  overfit gap            mean {np.mean(gaps):.3f}")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "best_config": configs[best], "best_err_acc": float(full[best]),
            "scene_ids": sids, "truth": truth.tolist(),
            "best_counts": M[best].tolist(),
            "null_mean": float(null.mean()), "null_p95": float(np.percentile(null, 95)),
            "holdout_mean": float(np.mean(te_s)), "overfit_gap": float(np.mean(gaps)),
        }, indent=1))
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
