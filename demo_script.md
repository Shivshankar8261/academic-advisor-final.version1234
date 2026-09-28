# Live Demo Script — VU AI Academic Advisor (~6 minutes)

**Before the talk:** start the backend (`cd backend && source .venv/bin/activate && uvicorn main:app --port 8000`)
and the frontend (`cd frontend && npm run dev`), open http://localhost:3000, and check the header says
"backend online". Keep strategy = **RAG + Student Profile (final)**. Click **New chat** between steps.

> The free Groq tier allows ~8k tokens/min per model. If one model is busy, the answer comes from the next model in
> the fallback chain automatically — point at the grey provider badge when that happens; it's a feature.

---

### Step 1 — Normal, grounded answer (30 s)
- Profile: **No profile**
- Type: `What minimum CGPA do I need to be promoted to the second year?`
- **Show:** answer "4.00", *high confidence*, *grounded* badge, expand **Sources** →
  *Student Handbook Aug 2026.pdf — Academic Regulations > 12. Progression (12.1)*. Point at the ~1–2 s timing
  and the retrieval time in milliseconds.

### Step 2 — Ambiguous question → the advisor asks (the brief's own example) (60 s)
- Profile: **No profile**
- Type: `Can I take Machine Learning next semester?`
- **Show:** the highlighted **"Question for you"** box asking for batch and prerequisites; the input box is
  focused and turns indigo.
- Reply: `I'm in the 2024 batch. I passed DATA202 Exploratory Data Analysis but failed MATH301 Optimization Techniques.`
- **Show:** final answer = *not eligible*, because MATH301 is a prerequisite for the 2024 batch — and ML (Semester 5,
  odd term) isn't in the next (even) term anyway. **Message:** asking one targeted question turned a guess
  into a correct, sourced recommendation.

### Step 3 — Same question, but with a structured student profile (45 s)
- Profile: **SYN-001** (2024 batch, failed Data Structures and MATH301)
- Type: `Can I register for Machine Learning this semester?`
- **Show:** no clarification needed — the profile supplies the facts; the answer names the failed prerequisite and
  cites *Course DATA301 Machine Learning* + *2024 Batch – Semester 5*.
- Optional follow-up: `I failed Data Structures. Can I re-take it this semester?` → not offered in the odd term
  (COMP201 is a Semester 2 course); options: re-register when offered / summer term if offered.

### Step 4 — Missing information → says so instead of inventing (30 s)
- Profile: **No profile**
- Type: `What is the exact date of the add/drop deadline this semester?`
- **Show:** amber banner *"I don't have enough information…"*; it still gives the rule it *does* know
  (within two weeks of commencement, clause 2.15) but not a made-up date.
- Backup: `What is DATA206? It is listed as a prerequisite for Machine Learning for the 2025 batch.`
  (DATA206 does not exist anywhere in the supplied data.)

### Step 5 — Conflicting rules (45 s)
- Profile: **No profile**
- Type: `I missed course registration because of a family function. Can I still register late?`
- **Show:** orange *conflicting rules* banner — Handbook 2.7 allows up to one calendar week (with a late fee),
  while the SOP says no late registration except medical exigencies/competitions; it tells the student to confirm
  with the Program Chair. Both sources cited.
- Backup: `How many University Core credits does the 2026 batch need?` → 24 in the semester spread vs 18 in the
  Struct_2026_DS summary sheet (a conflict inside the university's own spreadsheet).

### Step 6 — Why RAG matters: switch strategies live (60 s)
- Profile: **No profile**. Ask `What minimum attendance do I need to write the end-term exam?`
  1. Strategy **Baseline LLM** → generic "most universities…" answer, no sources.
  2. Strategy **Structured prompt (no docs)** → declines / low confidence (it's told not to guess, but has no data).
  3. Strategy **RAG** → "75%", cited to Handbook clause 7.2.
- **Message:** prompting alone makes the model *honest*; retrieval makes it *correct*; the profile makes it *personal*.

### Step 7 — Results slide (30 s)
Show the table from `eval/summary.md` (accuracy, hallucination rate, eligibility decisions, source correctness,
missing-info / conflict handling, response time) across Baseline → Structured → RAG → RAG + Profile.

---

**Likely questions & short answers**
- *Where do the student profiles come from?* Synthetic (SYN-001…010), generated from the real semester spread;
  credits/CGPA computed with handbook clauses 8.11/8.16. No real student data.
- *How do you know the current semester?* Derived: the offering file is Sept 2026 and handbook 1.2 says the odd
  semester runs Jul/Aug–Dec (stated as an assumption).
- *How is the eval scored?* Rule-based matching of flags + keywords against verified expected outcomes; limitation
  acknowledged, labels spot-checked by hand.
- *Why Groq + Gemini?* Lowest latency (~1–2 s end-to-end) with automatic fallback when the free-tier limit is hit.
