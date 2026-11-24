"""Phase 2 통과 조건: 40문장에 대해 '규칙만' / '규칙+LLM' 정확도 표 + 금지·불가능 문장에서 움직인(press) 개수.
사용법: python tests/eval_nlu.py [--modes rules rules+llm] [--testset nlu/testset_ko.yaml]
"""
import argparse, json, statistics, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml
from nlu.parse import parse_command

ap = argparse.ArgumentParser()
ap.add_argument("--testset", default=str(Path(__file__).resolve().parents[1] / "nlu" / "testset_ko.yaml"))
ap.add_argument("--modes", nargs="+", default=["rules", "rules+llm"])
ap.add_argument("--out", default=None, help="결과 JSON 저장 경로")
ap.add_argument("-v", "--verbose", action="store_true")
ap.add_argument("--scene", default="v2", choices=["v2", "v3"], help="v3: 6층 건물, 색 매핑 없음 (testset_v3_ko.yaml)")
a = ap.parse_args()
if a.scene == "v3" and a.testset.endswith("testset_ko.yaml"):
    a.testset = a.testset.replace("testset_ko.yaml", "testset_v3_ko.yaml")
FLOORS = (1, 2, 3, 4, 5, 6) if a.scene == "v3" else (1, 2, 3)
items = yaml.safe_load(open(a.testset, encoding="utf-8"))
report = {}
for mode in a.modes:
    rows, lat, fallback = [], [], 0
    for it in items:
        task, meta = parse_command(it["text"], mode, scene_floors=FLOORS, scene=a.scene)
        ok = task["intent"] == it["intent"] and (it["intent"] != "press" or task["floor"] == it.get("floor"))
        unsafe = it["category"] in ("forbidden", "impossible") and task["intent"] == "press"
        if meta.get("llm_ms") is not None and meta["parser"] == "rules+llm":
            lat.append(meta["llm_ms"])
        fallback += meta["parser"] == "rules(fallback)"
        rows.append({**it, "got": task, "ok": ok, "unsafe": unsafe, "parser": meta["parser"], "llm_ms": meta.get("llm_ms")})
        if a.verbose and not ok:
            print(f"  [{mode}] X {it['text']!r}: expect {it['intent']}/{it.get('floor')} got {task['intent']}/{task['floor']} ({meta['parser']})")
    cats = {}
    for r in rows:
        c = cats.setdefault(r["category"], [0, 0]); c[0] += r["ok"]; c[1] += 1
    report[mode] = {"correct": sum(r["ok"] for r in rows), "total": len(rows),
                    "unsafe_moves": sum(r["unsafe"] for r in rows), "by_category": cats,
                    "llm_ms_median": statistics.median(lat) if lat else None, "llm_fallbacks": fallback, "rows": rows}
print(f"{'mode':12s} {'정답':>7s} {'금지/불가 오동작':>14s}  " + "  ".join(f"{c:>11s}" for c in ("paraphrase", "forbidden", "impossible", "ambiguous")) + "  LLM중앙값  폴백")
for mode, r in report.items():
    cats = "  ".join(f"{r['by_category'].get(c, [0, 0])[0]:>5d}/{r['by_category'].get(c, [0, 0])[1]:<5d}" for c in ("paraphrase", "forbidden", "impossible", "ambiguous"))
    lm = f"{r['llm_ms_median']:.0f}ms" if r["llm_ms_median"] else "-"
    print(f"{mode:12s} {r['correct']:>3d}/{r['total']:<3d} {r['unsafe_moves']:>14d}  {cats}  {lm:>8s}  {r['llm_fallbacks']}")
if a.out:
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
