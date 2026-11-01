"""
The reported 5-turn chat against the real models, in every mode except the baseline control.

    python eval/run_memory_scenario.py                       # calls advisor.py directly
    python eval/run_memory_scenario.py https://...vercel.app # or a deployed API
"""
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

CHAT = ["BTech, 2024 batch, semester 5. Can I take a minor?", "What information do you need from me?",
        "Which batch and semester did I tell you?", "Correction: semester 3, not 5. Does that change your advice?",
        "Summarise what you know about me."]


def ask(url, text, strategy, history):
    if url:
        body = json.dumps({"message": text, "strategy": strategy, "conversation_history": history}).encode()
        req = urllib.request.Request(f"{url}/chat", body, {"content-type": "application/json"})
        return json.load(urllib.request.urlopen(req, timeout=90))
    import advisor
    return advisor.answer(text, strategy, None, history, guarded=True).model_dump()


def main():
    url = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else None
    for strategy in ("rag_structured", "agentic", "rag", "structured"):
        print(f"\n===== {strategy}")
        history = []
        for text in CHAT:
            r = ask(url, text, strategy, history)
            print(f"\nUSER: {text}\n[{r['provider']} | {r['usage']['total_tokens']} tokens | {r['confidence']} | "
                  f"grounded={r['grounded']} | sources={len(r['sources'])}]\n{r['answer'][:420]}")
            if r.get("clarifying_question"):
                print("ASKS:", r["clarifying_question"])
            history += [{"role": "user", "content": text},
                        {"role": "assistant", "content": r["answer"], "needs_clarification": r["needs_clarification"]}]


if __name__ == "__main__":
    main()
