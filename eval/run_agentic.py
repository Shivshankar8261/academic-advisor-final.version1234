"""
Phase 9 comparison: the RAG system (rag_structured) against the agentic system on every test case, same model,
same scorer as run_eval.py. Resumable: finished rows stay in agentic_raw.jsonl.

    LLM_PIN=groq:openai/gpt-oss-120b python eval/run_agentic.py
Writes eval/agentic_results.json (metrics plus every label).
"""
import json
import statistics
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR.parent / "backend"))
sys.path.insert(0, str(EVAL_DIR))

from run_eval import classify, run_case  # noqa: E402

STRATEGIES = ["rag_structured", "agentic"]
RAW = EVAL_DIR / "agentic_raw.jsonl"


def main():
    cases = {c["id"]: c for c in json.loads((EVAL_DIR / "test_cases.json").read_text())}
    done = {}
    if RAW.exists():
        for line in RAW.read_text().splitlines():
            r = json.loads(line)
            if r["response"].get("provider") not in (None, "none"):
                done[(r["id"], r["strategy"])] = r
    RAW.write_text("".join(json.dumps(r) + "\n" for r in done.values()))
    with open(RAW, "a") as out:
        for case in cases.values():
            for s in STRATEGIES:
                if (case["id"], s) in done:
                    continue
                row = run_case(case, s)
                done[(case["id"], s)] = row
                out.write(json.dumps(row) + "\n")
                out.flush()
                print(case["id"], s, row["response"].get("provider"), row["response"]["answer"][:70].replace("\n", " "),
                      flush=True)
    result = {"cases": len(cases), "metrics": {}, "rows": []}
    for s in STRATEGIES:
        rows = [done[(i, s)] for i in cases if (i, s) in done]
        verdicts = [(r, classify(cases[r["id"]], r["response"])) for r in rows]
        count = lambda label: sum(v["label"] == label for _, v in verdicts)  # noqa: E731
        elig = [v for _, v in verdicts if v["eligibility_correct"] is not None]
        src = [v for _, v in verdicts if v["source_correct"] is not None]
        real = [r["response"] for r in rows if r["response"].get("usage", {}).get("total_tokens")]
        result["metrics"][s] = {
            "n": len(rows), "correct": count("correct"), "partial": count("partially_correct"),
            "hallucination": count("unsupported_hallucination"), "incorrect": count("incorrect"),
            "eligibility_correct": sum(v["eligibility_correct"] for v in elig), "eligibility_n": len(elig),
            "source_correct": sum(v["source_correct"] for v in src), "source_n": len(src),
            "avg_time_s": round(statistics.mean(r["response"]["response_time_s"] for r in rows), 2),
            "median_time_s": round(statistics.median(r["response"]["response_time_s"] for r in rows), 2),
            "avg_tokens_llm_calls": round(statistics.mean(r["usage"]["total_tokens"] for r in real)) if real else 0,
            "answered_without_llm": len(rows) - len(real),
            "repairs": sum("Rejected the draft" in t["summary"] for r in rows
                           for t in r["response"].get("agent_trace", [])),
        }
        result["rows"] += [{"id": r["id"], "strategy": s, "label": v["label"], "provider": r["response"].get("provider"),
                            "answer": r["response"]["answer"]} for r, v in verdicts]
    (EVAL_DIR / "agentic_results.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(json.dumps(result["metrics"], indent=1))


if __name__ == "__main__":
    main()
