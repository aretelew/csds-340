"""Score every cached single model and every equal-weight percentile ensemble
of 2-4 members, then apply the agreed selection rule:

  1. eligible = AUC within 0.003 of the best mean AUC;
  2. among eligible, highest mean TPR@1%FPR;
  3. A beats B only if A's mean is higher AND A wins most paired repeats
     (harness.compare).

Also runs a split-half check of the selection itself: pick the winner using
the labels of a random half of the emails, score it on the other half, and
compare with fixed choices. Searching hundreds of combos on one dataset is
optimistic; this shows how much of the gain survives.

    uv run python -m experiments.final.report > experiments/final/report.txt
    ... --pool lr svc knn ... --vs lgbm+lr+mlp_bag5+spline_lr   (restricted pool)
"""
from itertools import combinations

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

from experiments.final import ensemble as E
from experiments.final.nested import CACHE
from experiments.harness import compare, tprAtFPR

AUC_TOL = 0.003
# never put two seed-variants of the same model in one ensemble
SAME = [{"hgb", "hgb_bag5"}, {"mlp", "mlp_bag5"}]


def pool():
    return sorted(p.stem for p in CACHE.glob("*.npz") if not p.stem.endswith("_outer"))


def combos(names, kmax=4):
    out = []
    for k in range(2, kmax + 1):
        for c in combinations(names, k):
            if not any(len(s & set(c)) > 1 for s in SAME):
                out.append(c)
    return out


def select(rows):
    best = max(r["auc"] for r in rows)
    elig = [r for r in rows if r["auc"] >= best - AUC_TOL]
    return best, sorted(elig, key=lambda r: -r["tpr1"])


def fmt(r):
    return (f"{r['name']:38s} AUC {r['auc']:.4f}±{r['auc_sd']:.4f}  "
            f"TPR@1% {r['tpr1']:.3f}±{r['tpr1_sd']:.3f}  pAUC.05 {r['pauc05']:.4f}")


def split_half(cands, fixed, B=40, seed=0):
    """cands: list of member tuples. Returns mean held-out TPR/AUC of the
    rule-selected candidate and of each fixed choice, over B random halves."""
    y = E.labels()
    S = {c: np.mean([E.pct_scores(n) for n in c], axis=0) for c in set(cands) | set(fixed)}
    sss = StratifiedShuffleSplit(n_splits=B, test_size=0.5, random_state=seed)
    res = {"selected": []} | {"+".join(f): [] for f in fixed}
    picks = []
    for a, b in sss.split(np.zeros(len(y)), y):
        def m(c, idx):
            s = S[c][:, idx]
            return (np.mean([roc_auc_score(y[idx], si) for si in s]),
                    np.mean([tprAtFPR(y[idx], si) for si in s]))
        on_a = {c: m(c, a) for c in cands}
        best = max(v[0] for v in on_a.values())
        pick = max((c for c in cands if on_a[c][0] >= best - AUC_TOL),
                   key=lambda c: on_a[c][1])
        picks.append("+".join(pick))
        res["selected"].append(m(pick, b))
        for f in fixed:
            res["+".join(f)].append(m(f, b))
    return {k: np.array(v) for k, v in res.items()}, picks


def main(names=None, vs=()):
    """names: the model pool (default: every cached model). vs: extra
    ensembles ("a+b+c", any cached models) to compare the winner against."""
    names = names or pool()
    print(f"pool: {', '.join(names)}\n")

    print("== single models: raw pooled OOF (as harness.evaluate) vs percentile-mapped ==")
    singles = []
    for n in names:
        r, p = E.raw(n), E.combo((n,))
        print(fmt(r))
        print(fmt(p | {"name": f"  mapped"}))
        singles.append(p)
    for n in sorted(p.stem[:-6] for p in CACHE.glob("*_outer.npz")):
        r = E.metrics(np.load(CACHE / f"{n}_outer.npz")["test"], E.labels(), n + " (outer only)")
        print(fmt(r))

    rows = singles + [E.combo(c) for c in combos(names)]
    best, elig = select(rows)
    print(f"\n== {len(rows)} candidates; best AUC {best:.4f}; "
          f"{len(elig)} within {AUC_TOL} -> ranked by TPR@1% ==")
    for r in elig[:30]:
        print(fmt(r))

    by = {r["name"]: r for r in rows}
    by |= {v: E.combo(tuple(v.split("+"))) for v in vs}
    top = elig[0]
    refs = list(vs) + [n for n in ["hgb", "hgb+et_nan", "mlp", "hgb+et_nan+mlp"]
                       if n in by and n not in vs]
    refs += [r["name"] for r in elig[1:6]]
    print(f"\n== paired compare (10 repeats): {top['name']} vs ==")
    for n in refs:
        for metric in ("tprs", "aucs"):
            d, w, beats = compare(top, by[n], metric)
            print(f"  {n:38s} {metric[:-1]:4s} diff {d:+.4f}  wins {w}/10  "
                  f"{'BEATS' if beats else '-'}")

    print("\n== split-half selection check (select on half A, score on half B, 40 halves) ==")
    cands = [tuple(r["name"].split("+")) for r in rows]
    every = tuple(n for n in names if n not in ("hgb_bag5", "mlp_bag5"))
    fixed = [c for c in [("hgb",), ("mlp",), ("hgb", "et_nan")] if c in cands]
    fixed += [every, tuple(top["name"].split("+"))]
    res, picks = split_half(cands, fixed)
    notes = {"+".join(every): " (all models, no selection)",
             top["name"]: " (picked on ALL data: biased)"}
    for k, v in res.items():
        print(f"  {k + notes.get(k, ''):58s} held-out AUC {v[:, 0].mean():.4f}  "
              f"TPR@1% {v[:, 1].mean():.3f}")
    d = res["selected"][:, 1] - res["hgb+et_nan"][:, 1] if "hgb+et_nan" in res else None
    if d is not None:
        print(f"  selected - hgb+et_nan TPR: {d.mean():+.3f} (better in {(d > 0).sum()}/40)")
    u, c = np.unique(picks, return_counts=True)
    print("  most picked:", ", ".join(f"{n} ({k})" for k, n in
                                     sorted(zip(c, u), reverse=True)[:6]))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", nargs="+", help="models to combine (default: all cached)")
    ap.add_argument("--vs", nargs="+", default=[], help='ensembles to compare against, e.g. "mlp+lr"')
    a = ap.parse_args()
    main(a.pool, a.vs)
