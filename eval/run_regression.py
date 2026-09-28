"""
Regression suite for the hardened advisor (cases R01-R14 in test_cases.json, plus any ids given).

Runs each case through the FINAL strategy (rag_structured) and scores it with the same rule-based
classifier as run_eval.py (so the labels mean the same thing). It does NOT touch the report files
(summary.md / results.csv / raw_results.jsonl); results go to eval/regression_results.json.

    python eval/run_regression.py            # all R* cases
    python eval/run_regression.py R11 R12    # selected cases
Exit code 1 if any case is not labelled "correct".
"""
import json
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR.parent / "backend"))
sys.path.insert(0, str(EVAL_DIR))

import advisor  # noqa: E402
from run_eval import classify  # noqa: E402


def main(ids: list[str]) -> int:
    cases = [c for c in json.loads((EVAL_DIR / "test_cases.json").read_text())
             if (c["id"] in ids if ids else c["id"].startswith("R"))]
    out, failed = [], 0
    for case in cases:
        resp = advisor.answer(case["query"], "rag_structured", case.get("student_id")).model_dump()
        verdict = classify(case, resp)
        ok = verdict["label"] == "correct"
        failed += not ok
        out.append({"id": case["id"], "query": case["query"], "expected": case["expected_outcome"],
                    "label": verdict["label"], "source_correct": verdict["source_correct"],
                    "provider": resp["provider"], "answer": resp["answer"],
                    "sources": [f"{s['source_file']} | {s['section']}" for s in resp["sources"]],
                    "flags": {k: resp[k] for k in ("insufficient_information", "conflict_detected",
                                                   "needs_clarification", "grounded")}})
        print(f"\n[{'PASS' if ok else 'FAIL'}] {case['id']} ({verdict['label']}, source ok: "
              f"{verdict['source_correct']}) via {resp['provider']}")
        print(f"  Q: {case['query']}")
        print(f"  expected: {case['expected_outcome']}")
        print("  answer: " + resp["answer"].replace("\n", "\n          "))
        for s in resp["sources"]:
            print(f"  source: {s['source_file']} | {s['section']}")
    (EVAL_DIR / "regression_results.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"\n{len(cases) - failed}/{len(cases)} regression cases passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
