"""Re-run the checks on saved sessions. DIR holds one folder per label: real/, print/, replay/, ..."""
import json
import sys
from pathlib import Path

from app import server
from app.pipeline import Landmarks, Passive


def run(root):
    if not server.analyzers:
        server.analyzers.update(passive=Passive(), landmarks=Landmarks())
    rows = []
    for meta in sorted(Path(root).glob("*/*/session.json")):
        s = json.loads(meta.read_text())
        images = [server.load_frame(p.read_bytes()) for p in sorted(meta.parent.glob("*.jpg"))]
        result, _ = server.analyze(s["actions"], images, server.to_seconds(s["times"]))
        rows.append((meta.parent.parent.name, meta.parent.name, result))
    return rows


def pct(n, total):
    return f"{100 * n / total:.0f}%"


def report(rows):
    for label, name, r in rows:
        print(f"{label:8} {name:24} {r['decision']:9} passive={r['scores']['passive']}  {','.join(r['reasons']) or '-'}")
    print(f"\n{'label':8} {'n':>4} {'PASS':>5} {'UNCERT':>6} {'FAIL':>5}  {'passive min/mid/max':>20}  result")
    for label in sorted({row[0] for row in rows}):
        results = [r for lbl, _, r in rows if lbl == label]
        n = len(results)
        count = {d: sum(r["decision"] == d for r in results) for d in ("PASS", "UNCERTAIN", "FAIL")}
        scores = sorted(r["scores"]["passive"] for r in results if r["scores"]["passive"] is not None)
        spread = f"{scores[0]:.2f}/{scores[len(scores) // 2]:.2f}/{scores[-1]:.2f}" if scores else "-"
        # real: share not passed (BPCER); attacks: share passed (APCER)
        result = f"BPCER {pct(n - count['PASS'], n)}" if label == "real" else f"APCER {pct(count['PASS'], n)}"
        print(f"{label:8} {n:>4} {count['PASS']:>5} {count['UNCERTAIN']:>6} {count['FAIL']:>5}  {spread:>20}  {result}")
    ms = sorted(r["ms"] for _, _, r in rows)
    print(f"\ntime per check: p50 {ms[len(ms) // 2]} ms, p95 {ms[int(len(ms) * 0.95)]} ms")


if __name__ == "__main__":
    rows = run(sys.argv[1])
    if rows:
        report(rows)
    else:
        print(f"no sessions in {sys.argv[1]} (expected {sys.argv[1]}/<label>/<session>/session.json)")
