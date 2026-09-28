"""
Evaluation harness: runs every test case through all four strategies and scores them.

    python eval/run_eval.py                      # all strategies, all cases
    python eval/run_eval.py --strategies rag     # subset
    python eval/run_eval.py --rescore            # recompute metrics from eval/raw_results.jsonl (no API calls)

Calls the backend functions directly (faster than HTTP; same code path as /chat).

SCORING (rule-based — a deliberate SIMPLIFICATION, report it as a limitation):
Behaviour detection uses the structured flags when present, OR keyword phrases in the answer
text (so the baseline, which has no flags, is judged on what it actually says):
  clarify       needs_clarification, or phrases like "which batch", "could you tell me"
  insufficient  insufficient_information, or phrases like "not specified", "do not have"
  conflict      conflict_detected, or phrases like "conflict", "inconsistent", "differ"
Eligibility decision is read from the answer text ("not eligible", "cannot", ... vs "yes", "eligible", ...).
Facts: each `required_facts` group must have at least one keyword present in the answer.
Forbidden facts (regression cases R*): if any `forbidden_facts` phrase appears (e.g. "Mysore"), the case is
labelled unsupported_hallucination regardless of anything else.

Per-case label:
  answer cases (factual / eligibility / unavailable):
      correct                  all fact groups present AND eligibility decision right (if applicable)
      incorrect                wrong eligibility decision, or it abstained (asked / said "insufficient")
      unsupported_hallucination  asserted an answer with none of the required facts (made-up content)
      partially_correct        anything in between (some facts, or right decision with missing facts)
  clarify / insufficient cases:
      correct                  expected behaviour detected
      partially_correct        the other uncertainty behaviour detected (asked instead of saying insufficient, or vice-versa)
      unsupported_hallucination  confidently answered anyway
  conflict cases:
      correct                  conflict (or an accepted alternative behaviour) detected AND required facts present
      partially_correct        only one of the two
      incorrect                neither (presented a single rule as if it were the only one)
Keyword matching can mis-label paraphrases, so labels were spot-checked by hand (see summary.md note).
"""
import argparse
import csv
import json
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR.parent / "backend"))

import advisor  # noqa: E402

STRATEGIES = ["baseline", "structured", "rag", "rag_structured"]
LABELS = ["correct", "partially_correct", "unsupported_hallucination", "incorrect"]

CLARIFY_PHRASES = ["which batch", "what batch", "could you tell", "can you tell", "could you confirm", "can you confirm",
                   "please let me know", "please share", "please tell", "which courses have you", "what courses have you",
                   "have you completed", "have you passed", "which year", "what year are you", "could you share",
                   "let me know your", "need to know"]
INSUFFICIENT_PHRASES = ["not specified", "not provided", "not available in", "do not have", "don't have",
                        "does not appear", "doesn't appear", "not mentioned", "no information", "cannot determine",
                        "can't determine", "cannot confirm", "can't confirm", "not included", "not stated",
                        "not listed", "unable to", "isn't specified", "is not defined", "not found", "not in the",
                        "don't know", "do not know", "no details", "not covered", "not defined", "do not appear",
                        "details are unknown", "is unknown", "are unknown", "isn't in the provided"]
CONFLICT_PHRASES = ["conflict", "inconsisten", "discrepan", "contradict", "differ", "two different", "disagree",
                    "however, the sop", "however, the handbook", "on the other hand"]
NOT_ELIGIBLE_PHRASES = ["not eligible", "ineligible", "cannot", "can't", "not able", "not allowed", "not permitted",
                        "not be permitted", "not offered", "no,", "no.", "unfortunately", "will not count",
                        "won't count", "not all", "only up to", "not be able", "isn't offered", "not available this"]
ELIGIBLE_PHRASES = ["you are eligible", "yes", "you can", "on track", "you meet", "you satisfy", "eligible to",
                    "you qualify", "you are allowed", "which meets the", "meets the requirement",
                   "meet the requirement", "satisfies the"]


def _plain(text: str) -> str:
    """Models emit non-breaking hyphens/spaces (e.g. 'one‑week', '75 %'); match them as ordinary spaces."""
    return re.sub(r"[‐‑‒–—  \-]+", " ", text.lower())


def has_any(text: str, phrases) -> bool:
    t = _plain(text)
    return any(_plain(p) in t for p in phrases)


def detect(resp: dict) -> dict:
    text = resp["answer"] + " " + (resp.get("clarifying_question") or "")
    return {
        "clarify": resp["needs_clarification"] or (("?" in text) and has_any(text, CLARIFY_PHRASES)),
        "insufficient": resp["insufficient_information"] or has_any(text, INSUFFICIENT_PHRASES),
        "conflict": resp.get("conflict_detected", False) or has_any(text, CONFLICT_PHRASES),
    }


def eligibility_decision(text: str) -> str | None:
    t = text.lower()
    if has_any(t, NOT_ELIGIBLE_PHRASES):
        return "not_eligible"
    if has_any(t, ELIGIBLE_PHRASES):
        return "eligible"
    return None


def facts_score(text: str, groups) -> float:
    if not groups:
        return 1.0
    return sum(has_any(text, g) for g in groups) / len(groups)


def source_correct(resp: dict, expected) -> bool | None:
    if not expected:
        return None
    cited = " ".join(f"{s['source_file']} {s['section']}" for s in resp["sources"]).lower()
    return any(e.lower() in cited for e in expected)


def classify(case: dict, resp: dict) -> dict:
    text = resp["answer"] + " " + (resp.get("clarifying_question") or "")
    beh = detect(resp)
    fs = facts_score(text, case.get("required_facts"))
    exp_elig = case.get("expected_eligibility")
    decision = eligibility_decision(resp["answer"]) if exp_elig else None
    elig_ok = (decision == exp_elig) if exp_elig else None
    expected = case["expected_behavior"]

    if expected == "answer":
        abstained = (beh["clarify"] or resp["insufficient_information"]) and fs == 0
        if exp_elig and decision and decision != exp_elig:
            label = "incorrect"
        elif abstained:
            label = "incorrect"
        elif fs == 1 and (elig_ok in (True, None)):
            label = "correct"
        elif fs == 0 and not elig_ok:
            label = "unsupported_hallucination"
        else:
            label = "partially_correct"
    elif expected in ("clarify", "insufficient"):
        other = "insufficient" if expected == "clarify" else "clarify"
        if beh[expected]:
            label = "correct"
        elif beh[other]:
            label = "partially_correct"
        else:
            label = "unsupported_hallucination"
    else:  # conflict
        # e.g. X02 also accepts asking for the batch; "answer" alone never counts as flagging the conflict
        accepted = case.get("accept_behaviors", ["conflict"])
        flagged = any(beh[b] for b in accepted if b in beh)
        if flagged and fs == 1:
            label = "correct"
        elif flagged or fs == 1:
            label = "partially_correct"
        else:
            label = "incorrect"
    # regression cases: a claim the documents do not support (e.g. "Mysore") fails the case outright
    forbidden = [f for f in case.get("forbidden_facts", []) if f.lower() in text.lower()]
    if forbidden:
        label = "unsupported_hallucination"
    return {"label": label, "facts_score": round(fs, 2), "eligibility_decision": decision,
            "eligibility_correct": elig_ok, "source_correct": source_correct(resp, case.get("expected_sources")),
            **{f"detected_{k}": v for k, v in beh.items()}}


def run_case(case: dict, strategy: str) -> dict:
    t0 = time.perf_counter()
    try:
        resp = advisor.answer(case["query"], strategy, case.get("student_id")).model_dump()
    except Exception as e:  # keep going; record the failure
        resp = {"answer": f"ERROR: {e}", "confidence": "low", "grounded": False, "sources": [],
                "needs_clarification": False, "clarifying_question": None, "insufficient_information": False,
                "conflict_detected": False, "response_time_s": time.perf_counter() - t0}
    row = {"id": case["id"], "category": case["category"], "strategy": strategy, "response": resp}
    # Multi-turn: answer the clarifying question and score the final recommendation.
    fu = case.get("follow_up")
    if fu:
        history = [{"role": "user", "content": case["query"]},
                   {"role": "assistant", "content": resp["answer"] + " " + (resp.get("clarifying_question") or ""),
                    "needs_clarification": resp["needs_clarification"]}]
        try:
            resp2 = advisor.answer(fu["message"], strategy, case.get("student_id"), history).model_dump()
        except Exception as e:
            resp2 = {"answer": f"ERROR: {e}", "sources": [], "needs_clarification": False,
                     "insufficient_information": False, "clarifying_question": None}
        row["follow_up_response"] = resp2
    return row


def score(rows: list[dict], cases: dict) -> list[dict]:
    scored = []
    for r in rows:
        case = cases[r["id"]]
        s = classify(case, r["response"])
        rec = {"id": r["id"], "category": r["category"], "strategy": r["strategy"], **s,
               "response_time_s": r["response"].get("response_time_s"),
               "provider": r["response"].get("provider"),
               "confidence": r["response"].get("confidence"), "answer": r["response"]["answer"].replace("\n", " ")}
        if "follow_up_response" in r:
            fu = case["follow_up"]
            fu_case = {"expected_behavior": "answer", "required_facts": fu.get("required_facts"),
                       "expected_eligibility": fu.get("expected_eligibility")}
            rec["follow_up_label"] = classify(fu_case, r["follow_up_response"])["label"]
        scored.append(rec)
    return scored


def pct(n, d):
    return f"{100 * n / d:.0f}%" if d else "n/a"


SUMMARY_DATA: dict = {}  # filled by summarise(); written to summary.json for the report generator


def summarise(scored: list[dict]) -> str:
    lines = ["# Evaluation summary — Basic LLM → Structured Prompting → RAG → RAG + Structured Student Data\n",
             f"Test cases: {len({r['id'] for r in scored})} | LLM (pinned for all strategies): "
             f"{', '.join(sorted({r['provider'] for r in scored if r.get('provider')}))} "
             f"(reasoning={advisor.llm.GROQ_REASONING_EFFORT}) | Scoring: rule-based (see eval/run_eval.py docstring)\n"]
    header = ["Metric"] + STRATEGIES
    table = {h: [] for h in header[1:]}
    metric_names = ["Accuracy (strictly correct)", "Accuracy incl. partial (×0.5)", "Hallucination rate",
                    "Incorrect responses", "Correct eligibility decisions", "Source/evidence correctness",
                    "Missing-info handling", "Conflicting-rules handling", "Clarification (ambiguous) handling",
                    "Final answer correct after follow-up", "Avg response time (s)"]
    for s in STRATEGIES:
        rs = [r for r in scored if r["strategy"] == s]
        n = len(rs)
        c = sum(r["label"] == "correct" for r in rs)
        p = sum(r["label"] == "partially_correct" for r in rs)
        h = sum(r["label"] == "unsupported_hallucination" for r in rs)
        inc = sum(r["label"] == "incorrect" for r in rs)
        el = [r for r in rs if r["eligibility_correct"] is not None]
        src = [r for r in rs if r["source_correct"] is not None]
        cat = lambda k: [r for r in rs if r["category"] == k]  # noqa: E731
        ok = lambda lst: pct(sum(r["label"] == "correct" for r in lst), len(lst))  # noqa: E731
        fu = [r for r in rs if "follow_up_label" in r]
        times = [r["response_time_s"] for r in rs if r["response_time_s"]]
        table[s] = [pct(c, n), pct(c + 0.5 * p, n), pct(h, n), f"{inc}/{n}",
                    pct(sum(r["eligibility_correct"] for r in el), len(el)),
                    pct(sum(r["source_correct"] for r in src), len(src)),
                    ok(cat("insufficient_info")), ok(cat("conflicting_rules")), ok(cat("ambiguous")),
                    pct(sum(r["follow_up_label"] == "correct" for r in fu), len(fu)),
                    f"{statistics.mean(times):.1f}" if times else "n/a"]
    SUMMARY_DATA["metrics"] = [{"metric": m, **{s: table[s][i] for s in STRATEGIES}} for i, m in enumerate(metric_names)]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for i, m in enumerate(metric_names):
        lines.append(f"| {m} | " + " | ".join(table[s][i] for s in STRATEGIES) + " |")

    lines.append("\n## Label distribution per strategy\n")
    lines.append("| Strategy | " + " | ".join(LABELS) + " |")
    lines.append("|" + "---|" * (len(LABELS) + 1))
    for s in STRATEGIES:
        rs = [r for r in scored if r["strategy"] == s]
        lines.append(f"| {s} | " + " | ".join(str(sum(r['label'] == l for r in rs)) for l in LABELS) + " |")
        SUMMARY_DATA.setdefault("labels", {})[s] = {l: sum(r["label"] == l for r in rs) for l in LABELS}

    lines.append("\n## Correct answers by category\n")
    cats = sorted({r["category"] for r in scored})
    lines.append("| Category (n) | " + " | ".join(STRATEGIES) + " |")
    lines.append("|" + "---|" * (len(STRATEGIES) + 1))
    for k in cats:
        n = len({r["id"] for r in scored if r["category"] == k})
        vals = []
        for s in STRATEGIES:
            rs = [r for r in scored if r["strategy"] == s and r["category"] == k]
            vals.append(f"{sum(r['label'] == 'correct' for r in rs)}/{len(rs)}")
        lines.append(f"| {k} ({n}) | " + " | ".join(vals) + " |")
        SUMMARY_DATA.setdefault("categories", []).append({"category": k, "n": n, **dict(zip(STRATEGIES, vals))})

    lines.append("\n## Metric definitions\n")
    lines.append("- **Accuracy**: share of the 25 cases labelled `correct`.\n"
                 "- **Hallucination rate**: share labelled `unsupported_hallucination` (asserted an answer that the "
                 "data does not support, or answered when it should have asked / said it lacks information).\n"
                 "- **Incorrect responses**: wrong eligibility decision, a single rule presented where the sources "
                 "conflict, or abstaining when the answer was available.\n"
                 "- **Correct eligibility decisions**: eligible / not-eligible read from the answer text matches the "
                 "verified decision (cases with an expected decision).\n"
                 "- **Source correctness**: at least one cited source matches the expected document section "
                 "(strategies without retrieval cite nothing, so they score 0).\n"
                 "- **Missing-info / conflicting-rules / clarification handling**: share of those categories labelled correct.\n"
                 "- **Final answer correct after follow-up**: for the 3 ambiguous cases, the student answers the "
                 "clarifying question and the second response is scored.\n"
                 "- **Avg response time**: wall-clock seconds per call, including retrieval.")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategies", nargs="*", default=STRATEGIES)
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--rescore", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    cases = {c["id"]: c for c in json.loads((EVAL_DIR / "test_cases.json").read_text())}
    raw_path = EVAL_DIR / "raw_results.jsonl"
    if args.rescore:
        rows = [json.loads(line) for line in raw_path.read_text().splitlines()]
    else:
        # Resumable: every finished row is appended to raw_results.jsonl immediately, and rows that already
        # exist (and did not fail to reach a model) are skipped on the next run.
        done = {}
        if raw_path.exists():
            for line in raw_path.read_text().splitlines():
                r = json.loads(line)
                if r["response"].get("provider") not in (None, "none"):
                    done[(r["id"], r["strategy"])] = r
        todo = [(c, s) for c in cases.values() if not args.ids or c["id"] in args.ids
                for s in args.strategies if (c["id"], s) not in done]
        print(f"{len(done)} results already saved; running {len(todo)} case x strategy combinations ...")
        raw_path.write_text("".join(json.dumps(r) + "\n" for r in done.values()))
        with ThreadPoolExecutor(args.workers) as pool, open(raw_path, "a") as out:
            for i, row in enumerate(pool.map(lambda cs: run_case(*cs), todo), 1):
                out.write(json.dumps(row) + "\n")
                out.flush()
                print(f"  [{i}/{len(todo)}] {row['id']} {row['strategy']}: {row['response']['answer'][:70]!r}")
        rows = [json.loads(line) for line in raw_path.read_text().splitlines()]

    rows.sort(key=lambda r: (list(cases).index(r["id"]), STRATEGIES.index(r["strategy"])))
    scored = score(rows, cases)
    with open(EVAL_DIR / "results.csv", "w", newline="") as f:
        fields = list(dict.fromkeys(k for r in scored for k in r))
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(scored)
    summary = summarise(scored)
    SUMMARY_DATA["providers"] = sorted({r["provider"] for r in scored if r.get("provider")})
    (EVAL_DIR / "summary.json").write_text(json.dumps(SUMMARY_DATA, indent=1))
    (EVAL_DIR / "summary.md").write_text(summary)
    print("\n" + summary)


if __name__ == "__main__":
    main()
