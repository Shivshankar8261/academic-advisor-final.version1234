"""
Phase 8 evaluation against the synthetic media (backend/make_media.py). Writes eval/multimodal_results.json.

  voice in     word error rate and key-term survival for Whisper
  transcript   code + grade precision and recall, whether grades in the image were read rather than obeyed
  voice out    round trip: Gemini TTS -> Whisper -> word error rate
  infographic  the credits drawn must equal the credits in the database, for every batch
"""
import json
import re
import sys
import time
import wave
from io import BytesIO
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
ROOT = EVAL_DIR.parent
sys.path.insert(0, str(ROOT / "backend"))

import multimodal  # noqa: E402
import structured  # noqa: E402

MEDIA = ROOT / "data/synthetic/multimodal"


def words(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9 ]", " ", text.lower().replace("-", " ")).split()


def wer(ref: str, hyp: str) -> float:
    r, h = words(ref), words(hyp)
    dist = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        prev, dist[0] = dist[0], i
        for j, hw in enumerate(h, 1):
            prev, dist[j] = dist[j], min(dist[j] + 1, dist[j - 1] + 1, prev + (rw != hw))
    return dist[-1] / max(len(r), 1)


def main():
    answers = json.loads((MEDIA / "answers.json").read_text())
    out = {"voice_in": [], "transcripts": {}, "voice_out": [], "infographic": {}}
    for a in answers["audio"]:
        t = time.perf_counter()
        text = multimodal.transcribe((MEDIA / "audio" / a["file"]).read_bytes(), a["file"])["text"]
        low = " ".join(words(text))
        out["voice_in"].append({"voice": a["voice"], "said": a["text"], "heard": text, "wer": round(wer(a["text"], text), 3),
                                "keys_kept": all(" ".join(words(k)) in low.replace("c g p a", "cgpa") for k in a["keys"]),
                                "seconds": round(time.perf_counter() - t, 2)})
    for name, want in answers["transcripts"].items():
        t = time.perf_counter()
        got = multimodal.read_transcript((MEDIA / "transcripts" / f"{name}.png").read_bytes())
        row = {"seconds": round(time.perf_counter() - t, 2), "provider": got["provider"], "readable": got["readable"]}
        if "courses" in want:
            found = {c["code"]: c["grade"] for c in got["courses"]}
            right = sum(found.get(k) == v for k, v in want["courses"].items())
            row.update(expected=len(want["courses"]), extracted=len(found), correct=right,
                       precision=round(right / max(len(found), 1), 3), recall=round(right / len(want["courses"]), 3),
                       batch_ok=got["batch"] == want["batch"], semester_ok=got["current_semester"] == want["current_semester"],
                       rejected=got["rejected"], wrong=[(k, v, found.get(k)) for k, v in want["courses"].items()
                                                        if found.get(k) != v])
        else:
            row["correctly_refused"] = not got["readable"] and not got["courses"]
        out["transcripts"][name] = row
        print(name, row, flush=True)
    for a in answers["audio"][:3]:
        t = time.perf_counter()
        audio, provider = multimodal.speak(a["text"])
        seconds = time.perf_counter() - t
        with wave.open(BytesIO(audio)) as w:
            duration = w.getnframes() / w.getframerate()
        heard = multimodal.transcribe(audio, "tts.wav")["text"]
        out["voice_out"].append({"provider": provider, "text": a["text"], "heard": heard, "wer": round(wer(a["text"], heard), 3),
                                 "audio_seconds": round(duration, 1), "generation_seconds": round(seconds, 1)})
        print(out["voice_out"][-1], flush=True)
    for batch in ("2022", "2023", "2024", "2025", "2026"):
        svg = multimodal.infographic_svg(batch)
        with structured._db() as con:
            db = con.execute("SELECT SUM(credits) FROM courses WHERE kind='programme' AND batch=? AND choice_of IS NULL",
                             (batch,)).fetchone()[0]
            count = con.execute("SELECT COUNT(*) FROM courses WHERE kind='programme' AND batch=? AND choice_of IS NULL",
                                (batch,)).fetchone()[0]
        drawn = sum(float(m) for m in re.findall(r": ([\d.]+) credits</text>", svg))
        out["infographic"][batch] = {"credits_drawn": drawn, "credits_in_db": db, "courses_in_db": count,
                                     "chips_drawn": svg.count("<g><title>"), "match": drawn == db and svg.count("<g><title>") == count}
    (EVAL_DIR / "multimodal_results.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps({"voice_in": out["voice_in"], "infographic": out["infographic"]}, indent=1))


if __name__ == "__main__":
    main()
