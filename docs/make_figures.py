"""Draws the figures embedded in the project document (run with any Python that has matplotlib)."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

INK = "#1f2937"
MUTED = "#6b7280"
NAVY = "#1e2a78"
COLORS = {  # fill, edge
    "raw": ("#fde8e8", "#b91c1c"),
    "proc": ("#fef3c7", "#b45309"),
    "store": ("#e0e7ff", "#3730a3"),
    "app": ("#dcfce7", "#15803d"),
    "llm": ("#f3e8ff", "#7e22ce"),
    "eval": ("#e0f2fe", "#0369a1"),
}


def box(ax, x, y, w, h, title, lines=(), kind="proc", title_size=10.5):
    fill, edge = COLORS[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                facecolor=fill, edgecolor=edge, linewidth=1.4))
    ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="top", fontsize=title_size, fontweight="bold", color=INK)
    for i, line in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.58 - i * 0.29, line, ha="center", va="top", fontsize=8.3, color=INK)


def arrow(ax, x1, y1, x2, y2, label=None, color=INK, style="-|>", ls="-", rad=0.0, label_dy=0.14):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=13, color=color,
                                 linewidth=1.3, linestyle=ls, connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + label_dy, label, ha="center", va="bottom", fontsize=7.8,
                color=MUTED, style="italic")


def path(ax, pts, label=None, color=INK, ls="-", label_at=0, label_dy=0.08):
    """Orthogonal connector through the given points, arrow head at the end."""
    xs, ys = zip(*pts)
    ax.plot(xs[:-1] + (xs[-1],), ys[:-1] + (ys[-1],), color=color, linewidth=1.3, linestyle=ls)
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=13, color=color, linewidth=1.3))
    if label:
        (x1, y1), (x2, y2) = pts[label_at], pts[label_at + 1]
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + label_dy, label, ha="center", va="bottom", fontsize=7.8,
                color=MUTED, style="italic", backgroundcolor="white")


def lane(ax, y, h, label):
    ax.add_patch(FancyBboxPatch((0.15, y), 19.7, h, boxstyle="round,pad=0.02,rounding_size=0.2",
                                facecolor="#f9fafb", edgecolor="#d1d5db", linewidth=1, linestyle="--"))
    ax.text(0.35, y + h - 0.15, label, ha="left", va="top", fontsize=10, fontweight="bold", color=NAVY)


def pipeline():
    fig, ax = plt.subplots(figsize=(16, 11.2))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 14)
    ax.axis("off")
    ax.text(10, 13.75, "VU AI Academic Advisor — end-to-end pipeline", ha="center", va="top",
            fontsize=16, fontweight="bold", color=NAVY)

    # ---------------- Lane A: offline data preparation ----------------
    lane(ax, 8.75, 4.55, "A. Offline data preparation (run once)")
    box(ax, 0.5, 9.9, 3.6, 2.75, "University files (data/raw)",
        ["Student Handbook Aug 2026 (PDF)", "SOP for Students (PDF)", "Semester Spread Sept 2026",
         "(XLSX, 10 sheets)", "Minor Courses (XLSX, 7 minors)"], "raw")
    box(ax, 4.9, 9.9, 3.2, 2.75, "extract_data.py",
        ["PDF text -> clauses by heading", "strip PDF control chars", "XLSX grid read by column",
         "per-batch prerequisites", "grade table transcribed"], "proc")
    box(ax, 8.9, 9.9, 3.7, 2.75, "Processed docs",
        ["handbook.md (17 clauses)", "sop.md (11 topics)", "programme_structure.md", "structure_summary.md, minors.md",
         "course_catalogue.json (135)"], "proc")
    box(ax, 13.4, 9.9, 2.9, 2.75, "ingest.py",
        ["split on ## / ### headings", "sub-clause splitting", "1 chunk per course", "MiniLM-L6-v2 embeddings",
         "(local, CPU)"], "proc")
    box(ax, 17.0, 10.55, 2.6, 2.1, "ChromaDB",
        ["412 chunks", "metadata: source_file,", "section, course_code"], "store")
    box(ax, 8.9, 8.95, 3.7, 0.8, "make_profiles.py -> 10 synthetic students", [], "store", title_size=8.3)
    arrow(ax, 4.1, 11.3, 4.9, 11.3)
    arrow(ax, 8.1, 11.3, 8.9, 11.3)
    arrow(ax, 12.6, 11.3, 13.4, 11.3)
    arrow(ax, 16.3, 11.6, 17.0, 11.6, "embed")
    arrow(ax, 10.75, 9.9, 10.75, 9.75)

    # ---------------- Lane B: online question answering ----------------
    lane(ax, 3.1, 5.15, "B. Online: answering a student question (~1-2 s)")
    box(ax, 0.5, 3.8, 3.0, 3.7, "Next.js chat UI",
        ["question + strategy", "+ optional student profile", "+ conversation history", "shows: answer, confidence,",
         "sources, clarifying question,", "missing-info / conflict banners"], "app")
    box(ax, 4.3, 3.8, 2.7, 3.7, "FastAPI /chat",
        ["advisor.py", "strategy router", "loads profile", "multi-turn merge:", "follow-up answer +",
         "original question"], "app")
    box(ax, 7.8, 5.75, 4.3, 1.75, "Retrieval (retrieval.py)",
        ["semantic top-5 + exact course-code/title", "lookup + prerequisite chain + student's",
         "current/next semester (rag_structured)"], "store", title_size=10)
    box(ax, 7.8, 3.8, 4.3, 1.7, "Prompt builder (prompts.py)",
        ["baseline | structured | rag | rag_structured", "role, constraints, <delimiters>, JSON spec,",
         "context chunks, profile, eligibility steps"], "app", title_size=10)
    box(ax, 12.9, 3.8, 3.4, 3.7, "LLM chain (llm.py)",
        ["1. Groq gpt-oss-120b", "2. Groq gpt-oss-20b", "3. Gemini flash-lite-latest", "4. Gemini 3.5-flash-lite",
         "strict JSON schema,", "429 -> cool-down & skip"], "llm")
    box(ax, 17.0, 3.8, 2.6, 3.7, "Validated JSON",
        ["answer, confidence,", "grounded, sources[],", "needs_clarification,", "clarifying_question,",
         "insufficient_information,", "conflict_detected"], "llm")
    arrow(ax, 3.5, 5.65, 4.3, 5.65, "POST")
    arrow(ax, 7.0, 6.6, 7.8, 6.6)
    arrow(ax, 9.95, 5.75, 9.95, 5.5)
    arrow(ax, 7.0, 4.65, 7.8, 4.65)
    arrow(ax, 12.1, 4.65, 12.9, 4.65)
    arrow(ax, 16.3, 5.65, 17.0, 5.65, "Pydantic")
    path(ax, [(18.3, 3.8), (18.3, 3.35), (2.0, 3.35), (2.0, 3.8)], "response rendered in the UI",
         color=COLORS["app"][1], label_at=1, label_dy=-0.3)
    path(ax, [(18.3, 10.55), (18.3, 8.45), (9.95, 8.45), (9.95, 7.5)], "top-k chunks + course records",
         color=COLORS["store"][1], ls="--", label_at=1, label_dy=-0.05)

    # ---------------- Lane C: evaluation ----------------
    lane(ax, 0.15, 2.2, "C. Evaluation harness")
    box(ax, 0.5, 0.35, 3.8, 1.6, "eval/test_cases.json",
        ["25 verified cases, 6 categories", "+ 3 multi-turn follow-ups"], "eval", title_size=10)
    box(ax, 5.3, 0.35, 5.2, 1.6, "eval/run_eval.py",
        ["each case x 4 strategies (same pinned model)", "rule-based labels: correct / partial /",
         ], "eval", title_size=10)
    ax.text(7.9, 0.72, "hallucinated / incorrect", ha="center", fontsize=8.3, color=INK)
    box(ax, 11.5, 0.35, 4.2, 1.6, "results.csv + summary.md",
        ["accuracy, hallucination, eligibility,", "sources, missing-info, conflicts, time"], "eval", title_size=10)
    box(ax, 16.6, 0.35, 3.0, 1.6, "Report & demo",
        ["Basic -> Structured ->", "RAG -> RAG + Profile"], "eval", title_size=10)
    arrow(ax, 4.3, 1.15, 5.3, 1.15)
    arrow(ax, 10.5, 1.15, 11.5, 1.15)
    arrow(ax, 15.7, 1.15, 16.6, 1.15)
    path(ax, [(7.9, 1.95), (7.9, 2.75), (5.65, 2.75), (5.65, 3.8)], "calls advisor.answer()",
         color=COLORS["eval"][1], ls="--", label_at=0, label_dy=-0.15)

    fig.savefig(OUT / "pipeline.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def strategies():
    fig, ax = plt.subplots(figsize=(12, 3.9))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3.9)
    ax.axis("off")
    ax.text(6, 3.85, "Four prompting strategies — each adds one layer", ha="center", va="top",
            fontsize=14, fontweight="bold", color=NAVY)
    steps = [
        ("1. baseline", ["raw question +", "generic instruction", "no role, no docs,", "free text"], "raw"),
        ("2. structured", ["+ role / persona", "+ constraints", "+ XML delimiters", "+ JSON output & flags"], "proc"),
        ("3. rag", ["+ top-k retrieved chunks", "+ cite source & section", "+ 'only use context'", "+ conflict rule"], "store"),
        ("4. rag_structured", ["+ student profile", "+ current-term context", "+ prerequisite chain", "+ eligibility steps"], "app"),
    ]
    for i, (title, lines, kind) in enumerate(steps):
        x = 0.3 + i * 2.95
        box(ax, x, 0.2 + i * 0.3, 2.6, 1.95, title, lines, kind, title_size=11)
        if i:
            arrow(ax, x - 0.35, 1.1 + i * 0.3, x, 1.2 + i * 0.3)
    fig.savefig(OUT / "strategies.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def results():
    """Grouped bars of the key eval metrics per strategy (ordinal blue ramp, validated with the dataviz validator)."""
    import json
    summary_path = OUT.parent.parent / "eval" / "summary.json"
    if not summary_path.exists():
        print("no eval/summary.json yet - skipping results chart")
        return
    data = json.loads(summary_path.read_text())
    wanted = ["Accuracy (strictly correct)", "Hallucination rate", "Correct eligibility decisions",
              "Source/evidence correctness", "Missing-info handling", "Conflicting-rules handling",
              "Clarification (ambiguous) handling"]
    short = ["Accuracy", "Hallucination\n(lower is better)", "Eligibility\ndecisions", "Source\ncorrectness",
             "Missing-info\nhandling", "Conflict\nhandling", "Clarification\nhandling"]
    rows = {m["metric"]: m for m in data["metrics"]}
    strategies_ = ["baseline", "structured", "rag", "rag_structured"]
    labels = ["Baseline LLM", "Structured prompt", "RAG", "RAG + Student profile"]
    colors = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # ordinal ramp, passes validate_palette --ordinal
    val = lambda m, s: float(str(rows[m][s]).rstrip("%")) if str(rows[m][s]).endswith("%") else 0.0  # noqa: E731

    fig, ax = plt.subplots(figsize=(12, 5.2))
    width = 0.2
    for j, s in enumerate(strategies_):
        xs = [i + (j - 1.5) * width for i in range(len(wanted))]
        ys = [val(m, s) for m in wanted]
        bars = ax.bar(xs, ys, width, color=colors[j], label=labels[j], edgecolor="white", linewidth=2)
        if s == "rag_structured":  # selective direct labels: final system only
            for b, y in zip(bars, ys):
                ax.text(b.get_x() + b.get_width() / 2, y + 1.5, f"{y:.0f}%", ha="center", va="bottom",
                        fontsize=8.5, color=INK)
    ax.set_xticks(range(len(wanted)))
    ax.set_xticklabels(short, fontsize=9, color=INK)
    ax.set_ylim(0, 110)
    ax.set_ylabel("% of relevant test cases", fontsize=9, color=MUTED)
    ax.yaxis.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#9ca3af")
    ax.tick_params(axis="y", colors=MUTED, labelsize=8.5, length=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=4, frameon=False, fontsize=9.5)
    ax.set_title("Evaluation results across development stages (25 test cases)", fontsize=12.5,
                 fontweight="bold", color=NAVY, pad=34)
    fig.savefig(OUT / "results.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    pipeline()
    strategies()
    results()
    print("figures written to", OUT)
