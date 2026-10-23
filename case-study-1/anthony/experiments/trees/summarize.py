"""Print all results (latest run of each name) sorted by AUC as a markdown table,
plus paired per-repeat differences vs a reference model."""
import json
import sys

from common import RES, load, paired

rows = {}
for line in open(RES):
    r = json.loads(line)
    rows[r["name"]] = r
rows = sorted(rows.values(), key=lambda r: -r["auc"])
print("| # | model | AUC | TPR@1% interp | step | pAUC.05 | fit s |")
print("|---|---|---|---|---|---|---|")
for i, r in enumerate(rows, 1):
    print(f"| {i} | {r['name']} | {r['auc']:.4f}±{r['auc_sd']:.4f} | "
          f"{r['tpr1']:.3f}±{r['tpr1_sd']:.3f} | {r['tpr1_step']:.3f} | "
          f"{r['pauc05']:.4f} | {r['fit_s']:.2f} |")
print(f"\n{len(rows)} configs")

ref = sys.argv[1] if len(sys.argv) > 1 else "HGB default (lr.1 it100 L31 msl20)"
_, y = load()
print(f"\nPaired diffs vs '{ref}' (mean±sd over the 5 repeats, same splits):")
for r in rows[:15]:
    if r["name"] == ref:
        continue
    da, sa, dt, st = paired(r["name"], ref, y)
    print(f"  {r['name']:42s} dAUC {da:+.4f}±{sa:.4f}  dTPR1 {dt:+.3f}±{st:.3f}")
