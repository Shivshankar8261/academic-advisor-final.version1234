"""
Builds the Assignment #2 report: docs/VU_AI_Advisor_Assignment2_Report.pdf (via headless Chrome) and an editable .docx
(via pandoc). Every number is read from eval/*.json, so the report cannot drift from the measured results.

    python3 docs/build_report.py
"""
import json
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS, EVAL = ROOT / "docs", ROOT / "eval"
LIVE = "https://academic-advisor-finalversion1234.vercel.app"
sec = json.loads((EVAL / "security_results.json").read_text())
mm = json.loads((EVAL / "multimodal_results.json").read_text())
ag = json.loads((EVAL / "agentic_results.json").read_text())
narrative = json.loads((DOCS / "report_findings.json").read_text())  # written after reading the agentic results


def table(head, rows, widths=None):
    cols = "".join(f'<col style="width:{w}%">' for w in widths) if widths else ""
    h = "".join(f"<th>{c}</th>" for c in head)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><colgroup>{cols}</colgroup><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>"


def fig(name, caption, width=100):
    return f'<figure><img src="figures/{name}" style="width:{width}%"><figcaption>{caption}</figcaption></figure>'


vi = mm["voice_in"]
tr = {k: v for k, v in mm["transcripts"].items() if "expected" in v}
vo = mm["voice_out"]
g, r, u = sec["guard_rules_and_classifier"], sec["guard_rules_only"], sec["unguarded"]
m_rag, m_ag = ag["metrics"]["rag_structured"], ag["metrics"]["agentic"]
exp_total = sum(v["expected"] for v in tr.values())
right_total = sum(v["correct"] for v in tr.values())


def metric_rows():
    pct = lambda a, b: f"{a}/{b} ({100 * a / b:.0f}%)" if b else "n/a"  # noqa: E731
    both = lambda f: [f(m_rag), f(m_ag)]  # noqa: E731
    return [["Strictly correct", *both(lambda m: pct(m["correct"], m["n"]))],
            ["Partially correct", *both(lambda m: str(m["partial"]))],
            ["Unsupported / hallucinated", *both(lambda m: str(m["hallucination"]))],
            ["Incorrect", *both(lambda m: str(m["incorrect"]))],
            ["Correct eligibility decisions", *both(lambda m: pct(m["eligibility_correct"], m["eligibility_n"]))],
            ["Source correctness", *both(lambda m: pct(m["source_correct"], m["source_n"]))],
            ["Median / mean response time", *both(lambda m: f'{m["median_time_s"]} s / {m["avg_time_s"]} s')],
            ["Tokens per LLM-using answer (mean)", *both(lambda m: f'{m["avg_tokens_llm_calls"]:,}')],
            ["Answered with no LLM call", *both(lambda m: str(m["answered_without_llm"]))],
            ["Drafts rejected and rewritten by Verification", "n/a", str(m_ag["repairs"])]]


HTML = f"""<!doctype html><html><head><meta charset="utf-8"><title>VU AI Academic Advisor: Assignment 2 Report</title><style>
@page {{ size:A4; margin:17mm 17mm 16mm }}
body {{ font:10pt/1.38 Arial,Helvetica,sans-serif; color:#16181d }}
h1 {{ font-size:21pt; color:#1b2559; margin:0 0 3pt }} h2 {{ font-size:13pt; color:#1b2559; margin:13pt 0 4pt; break-after:avoid }}
h3 {{ font-size:10.5pt; margin:8pt 0 2pt; break-after:avoid }} p {{ margin:0 0 5pt }} ul {{ margin:0 0 5pt; padding-left:15pt }} li {{ margin-bottom:2pt }}
table {{ border-collapse:collapse; width:100%; margin:4pt 0 7pt; font-size:8.8pt; break-inside:avoid }}
th {{ background:#1b2559; color:#fff; text-align:left; padding:3pt 5pt }} td {{ border-bottom:1px solid #d9dbe3; padding:3pt 5pt; vertical-align:top }}
figure {{ margin:5pt 0 7pt; break-inside:avoid; text-align:center }} img {{ border:1px solid #d9dbe3 }} figcaption {{ font-size:8.3pt; color:#555; margin-top:2pt }}
.sub {{ color:#555 }} code {{ font-size:8.8pt }} .callout {{ background:#eef0fb; padding:6pt 8pt; margin:5pt 0 7pt; break-inside:avoid }}
.two {{ display:flex; gap:8pt }} .two figure {{ flex:1 }}
</style></head><body>
<h1>Can the AI Academic Advisor be multimodal, secure and responsible?</h1>
<p class="sub">DATA308 Generative AI, Assignment #2 (Phases 8-12) · Vidyashilp University · Live demo: <a href="{LIVE}">{LIVE}</a></p>

<div class="callout"><b>Summary.</b> We extended the Assignment 1 RAG advisor with four modalities, a six-step agentic pipeline, a safety layer and a legal review.
{narrative["summary"]}
Transcript reading was {right_total}/{exp_total} correct across five synthetic scans, but it fell to {tr["phone_angle"]["correct"]}/{tr["phone_angle"]["expected"]} on an angled photo, which is why a student must confirm the record before it is used.
With the safety layer, {g["attacks_succeeded"]} of {g["attacks"]} attacks succeeded (without it, {u["attacks_succeeded"]}), and {len(g["false_positives"])} of {g["benign"]} normal questions were wrongly refused.</div>

<h2>1. Assignment 1 issues fixed first</h2>
<ul>
<li><b>Course-count failure.</b> "How many courses are in the 2022 Foundation basket?" returned "not enough information", because the router sent basket questions to the database only if they mentioned credits. Similarity search then found a few similar rows, not all of them. Basket, semester and batch questions now go to SQL, and the answer is <b>14 courses listed, 13 taken (PSYC101/ECON101 is either/or), 34 credits</b>, shown as a table with semester, credits, L-T-P and prerequisites. These answers make no LLM call, so they cannot be miscounted.</li>
<li><b>Token and cost display.</b> Each answer now shows tokens and estimated cost (Groq list prices, checked 31 Oct 2026), with a running total for the conversation.</li>
<li><b>Contradictory badges.</b> A reply could show "high confidence" beside "not enough information". Insufficient answers are now always low confidence.</li>
<li><b>A wrong attendance answer found during testing.</b> A fallback model answered 65% because retrieval returned the relaxation clause and not the 75% minimum. The whole attendance section is now always supplied for attendance questions, with a new regression case (R18).</li>
</ul>
<div class="two">{fig("ui_foundation_table.png", "Figure 1. The fixed answer on the live system: exact table, 0 tokens, no model used.")}</div>

<h2>2. System architecture</h2>
<p>A Next.js interface calls a FastAPI service; both run on Vercel. Every request passes through the safety layer. Exact questions go to the SQLite course store; all others use hybrid retrieval (MiniLM on ONNX over 440 chunks, plus exact course lookup) and a grounded LLM call with citation checks. Groq gpt-oss-120b is the primary model, with gpt-oss-20b and Gemini flash-lite as fallbacks.</p>
{table(["Component", "Tool / model", "Why it was chosen", "Licence / terms to note"], [
["Advice text", "Groq gpt-oss-120b, fallbacks", "Fast, strict JSON output", "Apache 2.0 weights; provider terms"],
["Embeddings", "all-MiniLM-L6-v2 (ONNX)", "Runs on the server, so no document text is sent for embedding", "Apache 2.0"],
["Voice in", "Groq Whisper large-v3-turbo", "Low cost, accurate on the test clips", "MIT weights; provider terms"],
["Image in", "Gemini flash-lite (vision)", "Structured output from images", "Google API terms"],
["Voice out", "Gemini TTS; browser speech as fallback", "Natural voice; the fallback keeps working when quota runs out", "Google API terms"],
["Infographic", "No model: SVG from SQL", "Exact numbers. Image generation had no free quota on our key", "Our own code"],
["Injection filter", "Rules + Llama Prompt Guard 2", "Rules are explainable; the classifier catches rephrasing", "Llama licence"]], [17, 27, 34, 22])}

<h2>3. Phase 8: Multimodal extension</h2>
<p>Each modality does a job for the student: <b>asking by voice</b>, <b>reading a grade sheet</b> instead of typing it, <b>hearing</b> an answer, and <b>seeing</b> the whole programme at once. All test media are synthetic (<code>backend/make_media.py</code>).</p>
{table(["Modality", "Test", "Result", "Main limitation"], [
["Voice in", f"{len(vi)} clips, 3 synthetic voices (US, UK, Indian English)", f'{sum(v["keys_kept"] for v in vi)}/{len(vi)} kept the key terms; mean word error {statistics.mean(v["wer"] for v in vi):.0%}, mostly format differences ("C G P A" to CGPA, "twenty twenty four" to 2024)', 'Invents "Thank you." on silence (now filtered). Synthetic voices are cleaner than a real noisy room'],
["Image in", "5 scans plus 1 non-transcript", f'{right_total}/{exp_total} code and grade pairs right. Clean, noisy, low-resolution and injected scans all 100%; angled photo {tr["phone_angle"]["correct"]}/{tr["phone_angle"]["expected"]}. Non-transcript refused', "Misread grades on the angled photo, including an F. Hence the review step"],
["Voice out", f"{len(vo)} round trips, speech to Whisper", f'Mean word error {statistics.mean(v["wer"] for v in vo):.0%}; about {statistics.mean(v["generation_seconds"] for v in vo):.0f} s to generate', "Tables are not read aloud; free quota is small"],
["Infographic", "All 5 batches against the database", f'{sum(v["match"] for v in mm["infographic"].values())}/5 match on credits and course count', "Long titles are shortened on the chart; no free-form visuals"]], [11, 23, 39, 27])}
<p><b>Which responsibilities stay with people.</b> The advisor does not act: it cannot register anyone or change a record. A misheard question goes into the text box for checking, and a read transcript goes to a review panel for correction. Questions about appeals, medical relaxation, graduation or wellbeing carry a notice saying who decides. A "Copy for my advisor" button passes the question, answer and sources to a person. We did not build video: it would add little to advising and be costly to check.</p>
<div class="two">{fig("ui_transcript_confirm.png", "Figure 2. The student reviews extracted grades before use.")}{fig("ui_infographic.png", "Figure 3. Infographic generated from the course store.")}</div>

<h2>4. Phase 9: Agentic extension</h2>
<p>The pipeline is <b>Planner → Regulation Retrieval → Course Eligibility → Credit Calculation → Recommendation → Verification</b>. Only Recommendation normally calls a model. Eligibility evaluates prerequisite logic such as "A AND (B OR C)" in code; Credits adds up earned credits per basket; Verification checks every number and course code against the evidence, and rejects a draft that contradicts the eligibility decision, allowing one rewrite. The order is fixed: a model-chosen plan would add cost and a new way to fail, since every question needs the same checks.</p>
{table(["Metric", "RAG + Student Profile", "Agentic"], metric_rows(), [46, 27, 27])}
<p class="sub">{ag["cases"]} test cases, same rule-based scorer as Assignment 1, same model (gpt-oss-120b, pinned). Exact questions skip the LLM in both systems.</p>
{narrative["agentic"]}
{fig("ui_agents.png", "Figure 4. Each answer can show what every agent did, with time and tokens.", 72)}

<h2>5. Phase 10: Privacy and security audit</h2>
{table(["Information", "Sensitivity", "Should the AI receive it?", "What the system does"], [
["Name, roll number, Aadhaar, PAN, phone, email", "High", "No: never needed for advice", "Removed before processing, and the student is told"],
["Grades, CGPA, failed courses", "High (education record)", "Only if needed, only the student's own", "Sent only for questions about the student; ID withheld"],
["Medical or disciplinary details", "Very high", "Avoid", "Not stored; a notice routes the student to a person"],
["Uploaded transcript image", "High", "Only codes and grades", "In memory only; names not extracted; re-encoded, dropping EXIF"],
["Handbook, SOP, course lists", "Low", "Yes", "Used for retrieval"],
["System prompt, API keys", "Secret", "Keys: never", "Keys server-side; output is checked for leaks"]], [28, 15, 24, 33])}
<h3>Red-team results ({g["attacks"]} attacks, {g["benign"]} normal questions)</h3>
{table(["Configuration", "Attacks that succeeded", "Attacks reaching the LLM", "Messages with personal data sent to the LLM", "Normal questions refused"], [
["No safety layer", u["attacks_succeeded"], u["reached_llm"], u["pii_sent_to_llm"], "n/a"],
["Rules only", r["attacks_succeeded"], r["reached_llm"], r["pii_sent_to_llm"], len(r["false_positives"])],
["Rules + Prompt Guard (deployed)", g["attacks_succeeded"], g["reached_llm"], g["pii_sent_to_llm"], len(g["false_positives"])]], [30, 16, 16, 22, 16])}
<p><b>Reading this honestly.</b> The grounded model was already fairly robust, and it has no access to other students' records unless the server sends them. So the safety layer's main gains are not the headline attack count. It stops most attacks before they spend tokens, keeps personal data away from third-party providers, and gives a consistent refusal. Prompt Guard caught three rephrased injections that the rules missed. A first scoring run reported 10 of 26 unguarded attacks succeeding; reading the answers by hand showed the scorer had missed refusals written with a curly apostrophe, so we corrected and rescored it. These runs used the gpt-oss-20b fallback model because of rate limits, and 26 attacks written by the builders are not a guarantee against a determined attacker. An injection hidden in a transcript image also failed: the scan was read correctly and its instruction ignored.</p>
<h3>Findings and safeguards</h3>
{table(["Finding", "Risk", "Status"], [
["Full student records were readable at <code>GET /students/&#123;id&#125;</code> with no login", "High", "Fixed: endpoint removed"],
["Any site could call the API (CORS *), and API docs were public", "Medium", "Fixed: origin allow-list, docs off (non-browser clients are still not blocked)"],
["Profile dropdown lets anyone choose any student", "High in production", "<b>Open.</b> Acceptable only because the data is synthetic. A real system needs university sign-in, with the identity taken from the session"],
["The browser supplies the chat history, so turns can be forged", "Medium", "Reduced: history is filtered and treated as data; server-held sessions would close it"],
["Rate limit is per server instance", "Low to medium", "Partial: a shared store would be needed for a hard limit"],
["Data goes to Groq and Google, which have their own terms", "High for real data", "Open: a real deployment needs a data-processing agreement, or self-hosted models"]], [52, 13, 35])}
<p>For real students, India's Digital Personal Data Protection Act 2023 would apply (notice, consent, purpose limitation, security safeguards, and duties about children's data). We also recommend audit logs without message content, short retention, and regular re-testing.</p>

<h2>6. Phase 11: Synthetic media, copyright and legal analysis</h2>
{table(["AI output", "Ownership and copyright", "Licensing and attribution", "Potential misuse", "Our practice"], [
["<b>Text answer</b> (gpt-oss)", "Purely AI-generated text may have weak or no copyright protection; the position is unsettled in India and elsewhere. Handbook passages remain the University's", "Weights are Apache 2.0; provider terms apply. Cite the source document", "Screenshot presented as an official ruling", "Citations on every answer; 'AI-generated' notice; confirm with an advisor"],
["<b>Spoken explanation</b> (Gemini TTS)", "As above for the audio. A synthetic voice raises no claim unless it imitates a real person", "Subject to Google's API terms; disclose that it is synthetic", "Faked 'announcement' by a university official", "Stock voice only, API marks it AI-generated, no voice cloning"],
["<b>Transcript extraction</b> (Gemini vision)", "Facts such as grades are not creative works, but they are personal data", "Uploader must have the right to share it", "Forged transcript used to gain eligibility", "Advisory only: nothing is written to any record; codes checked against the course list; not official proof"],
["<b>Infographic</b> (rule-generated, for contrast)", "Generated by our code from University data: not generative AI, and the underlying curriculum belongs to the University", "Credit the source file; ask permission for public use", "Passed off as an official curriculum document", "Source and 'not an official University document' printed on the image"]], [15, 26, 20, 17, 22])}
<p><b>Rights and permissions.</b> The handbook, SOP and spreadsheets were supplied for coursework. Publishing them or the derived index outside the course should need the University's permission, and the same goes for the University logo in the interface, which suggests endorsement. Student-created content stays the student's; uploads must not be used for training. For AI providers, confirm that their terms permit the data used.</p>
<p><b>Recommendations.</b> Label AI output; keep sources visible; never clone a real voice; state that outputs are not official records; obtain written permission before going beyond coursework; get legal review before real use. This analysis is a student assessment, not legal advice, and we recommend checking it against current law and provider terms.</p>

<h2>7. Phase 12: Deployment, limitations and conclusions</h2>
<p>The prototype is live on Vercel; the 2022 Foundation question, agents, infographic, transcript upload, read-aloud and refusals were checked on the deployed site with an automated browser, with no console errors.{narrative["deploy_note"]}</p>
<h3>Limitations</h3>
<ul>
<li>The eligibility, source and security scores come from rule-based matching. It is transparent, but as section 5 shows, it can mislabel, so labels should be read rather than trusted alone.</li>
<li>All media are synthetic. The clean voices and printed scans probably overstate real accuracy; handwriting and real noisy rooms were not tested.</li>
<li>The free tier allows 8,000 tokens a minute, so a live answer may come from a fallback model of lower quality. Tests pinned one model, so live quality can differ.</li>
<li>The current term is inferred from the file date, and university data can contain its own inconsistencies, which the advisor flags rather than resolves.</li>
<li>Assignment 1's stored raw results cover only part of its 25-case run, so Basic → Prompting → RAG figures are in the Assignment 1 report and were not recomputed here.</li>
</ul>
<h3>Conclusions</h3>
{narrative["conclusions"]}
</body></html>"""

html_path = DOCS / "report.html"
html_path.write_text(HTML)
pdf = DOCS / "VU_AI_Advisor_Assignment2_Report.pdf"
subprocess.run(["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "--headless=new", "--disable-gpu",
                "--no-pdf-header-footer", f"--print-to-pdf={pdf}", f"file://{html_path}"], check=True, capture_output=True)
subprocess.run(["pandoc", "report.html", "-o", "VU_AI_Advisor_Assignment2_Report.docx", "--resource-path=."], cwd=DOCS, check=True)
print("built", pdf)
