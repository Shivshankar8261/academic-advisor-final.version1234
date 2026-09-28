// Builds docs/VU_AI_Academic_Advisor_Documentation.docx
// Reads live project files (profiles, test cases, eval summary) so the numbers always match the code.
// Run:  node build_doc.js
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, AlignmentType, HeadingLevel,
  LevelFormat, WidthType, BorderStyle, ShadingType, PageBreak, TableOfContents, Header, Footer, PageNumber,
} = require("docx");

const ROOT = path.resolve(__dirname, "..");
const read = (p) => fs.readFileSync(path.join(ROOT, p));
const readJSON = (p) => JSON.parse(read(p).toString());
const exists = (p) => fs.existsSync(path.join(ROOT, p));

const profiles = readJSON("data/synthetic/student_profiles.json");
const cases = readJSON("eval/test_cases.json");
const summary = exists("eval/summary.json") ? readJSON("eval/summary.json") : null;

const NAVY = "1E2A78";
const FONT = "Arial";
const CONTENT_W = 9026; // A4 width 11906 - 2 x 1440 margins

// ---------- helpers ----------
const t = (text, o = {}) => new TextRun({ text, font: FONT, ...o });
const p = (text, o = {}) =>
  new Paragraph({ spacing: { after: 120, line: 300 }, ...o, children: Array.isArray(text) ? text : [t(text)] });
const rich = (parts) =>
  new Paragraph({
    spacing: { after: 120, line: 300 },
    children: parts.map((x) => (typeof x === "string" ? t(x) : t(x[0], x[1]))),
  });
const h1 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_1, pageBreakBefore: true, children: [t(text)] });
const h2 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [t(text)] });
const h3 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [t(text)] });
const bullet = (parts, level = 0) =>
  new Paragraph({
    numbering: { reference: "bullets", level },
    spacing: { after: 60, line: 280 },
    children: (Array.isArray(parts) ? parts : [parts]).map((x) => (typeof x === "string" ? t(x) : t(x[0], x[1]))),
  });
const numbered = (parts, ref = "steps") =>
  new Paragraph({
    numbering: { reference: ref, level: 0 },
    spacing: { after: 60, line: 280 },
    children: (Array.isArray(parts) ? parts : [parts]).map((x) => (typeof x === "string" ? t(x) : t(x[0], x[1]))),
  });
const code = (lines) =>
  lines.map(
    (line, i) =>
      new Paragraph({
        shading: { fill: "F3F4F6", type: ShadingType.CLEAR, color: "auto" },
        spacing: { after: i === lines.length - 1 ? 160 : 0, line: 250 },
        indent: { left: 200, right: 200 },
        children: [new TextRun({ text: line || " ", font: "Courier New", size: 17 })],
      }),
  );
const image = (file, widthPx, heightPx, caption) => [
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 120, after: 60 },
    children: [
      new ImageRun({
        type: "png",
        data: fs.readFileSync(path.join(__dirname, file)),
        transformation: { width: widthPx, height: heightPx },
        altText: { title: caption, description: caption, name: path.basename(file) },
      }),
    ],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 200 },
    children: [t(caption, { italics: true, size: 18, color: "555555" })],
  }),
];
const border = { style: BorderStyle.SINGLE, size: 4, color: "C9CED6" };
const borders = { top: border, bottom: border, left: border, right: border };
function table(headers, rows, widths) {
  const total = widths.reduce((a, b) => a + b, 0);
  const cell = (text, i, header) =>
    new TableCell({
      borders,
      width: { size: widths[i], type: WidthType.DXA },
      shading: header ? { fill: "E3E7F5", type: ShadingType.CLEAR, color: "auto" } : undefined,
      margins: { top: 60, bottom: 60, left: 100, right: 100 },
      children: String(text)
        .split("\n")
        .map((line) => new Paragraph({ children: [t(line, { bold: header, size: 18 })] })),
    });
  return new Table({
    width: { size: total, type: WidthType.DXA },
    columnWidths: widths,
    rows: [
      new TableRow({ tableHeader: true, children: headers.map((h, i) => cell(h, i, true)) }),
      ...rows.map((r) => new TableRow({ children: r.map((c, i) => cell(c ?? "", i, false)) })),
    ],
  });
}
const gap = () => new Paragraph({ spacing: { after: 120 }, children: [] });

// ---------- content ----------
const children = [];

// Title page
children.push(
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 1200, after: 400 },
    children: [
      new ImageRun({
        type: "png",
        data: read("frontend/public/vu-logo.png"),
        transformation: { width: 300, height: 109 },
        altText: { title: "Vidyashilp University logo", description: "Vidyashilp University logo", name: "logo" },
      }),
    ],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 200 },
    children: [t("VU AI Academic Advisor", { bold: true, size: 52, color: NAVY })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 600 },
    children: [
      t("Design, Build, Evaluation and Deployment of a RAG-based Academic Decision-Support System", {
        size: 28,
        color: "444444",
      }),
    ],
  }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [t("DATA308 Generative AI — Assignment #1", { size: 24 })] }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 1400 },
    children: [t("Technical documentation of the complete work", { size: 22, color: "666666" })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [
      t("Stack: Python · FastAPI · ChromaDB · sentence-transformers · Groq (gpt-oss) · Google Gemini · Next.js", {
        size: 18,
        color: "666666",
      }),
    ],
  }),
  new Paragraph({ children: [new PageBreak()] }),
  new Paragraph({ heading: HeadingLevel.HEADING_1, children: [t("Contents")] }),
  new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-2" }),
  p("Right-click the table of contents and choose “Update Field” in Word to refresh page numbers.", {
    children: [t("Right-click the table of contents and choose “Update Field” in Word to refresh page numbers.", { italics: true, size: 18, color: "777777" })],
  }),
);

// 1. Introduction
children.push(
  h1("1. Introduction"),
  p(
    "This document describes, end to end, how the VU AI Academic Advisor was designed, built and evaluated. The assignment asks for an advisor that “knows when to answer, what to answer, when to ask and when to say it does not have enough information”. The system answers B.Tech students' questions about courses, prerequisites, credits, attendance, registration, progression and graduation using only Vidyashilp University's own documents, cites the exact document and section it used, asks a follow-up question when it lacks facts about the student, and explicitly says so when the documents do not contain the answer or contain conflicting rules.",
  ),
  h2("1.1 Objectives"),
  bullet("Build a Retrieval-Augmented Generation (RAG) advisor grounded in the four university files supplied with the assignment."),
  bullet("Make the advisor interactive: ask targeted clarifying questions instead of assuming facts."),
  bullet("Detect missing information and conflicting rules, and never invent course codes, thresholds or dates."),
  bullet("Compare four development stages quantitatively: Basic LLM → Structured Prompting → RAG → RAG + Structured Student Data."),
  bullet("Deploy a working, low-latency prototype with a clear chat interface for a live demonstration."),
  h2("1.2 What was delivered"),
  table(
    ["Deliverable", "Where"],
    [
      ["Data pipeline that converts the raw PDFs/spreadsheets into citable documents", "backend/extract_data.py, data/processed/"],
      ["10 synthetic, anonymised student profiles", "backend/make_profiles.py, data/synthetic/"],
      ["Vector store (412 chunks) and hybrid retrieval", "backend/ingest.py, backend/retrieval.py"],
      ["Four prompting strategies", "backend/prompts.py"],
      ["LLM layer with low-latency fallback chain", "backend/llm.py"],
      ["REST API (FastAPI)", "backend/main.py, backend/advisor.py"],
      ["Chat web application (Next.js)", "frontend/app/page.tsx"],
      ["25-case evaluation set, runner and results", "eval/"],
      ["Demo script and this documentation", "demo_script.md, docs/"],
    ],
    [5400, 3626],
  ),
);

// 2. Data provided
children.push(
  h1("2. Source Data"),
  p(
    "No synthetic documents were created. Every rule, course, credit value and semester used by the advisor comes from the four files supplied with the assignment. The only synthetic data are the student profiles, which the assignment explicitly requires because no real student data is provided.",
  ),
  table(
    ["File", "Content", "How it is used"],
    [
      ["Student Handbook Aug 2026 (PDF, 92 pages)", "Academic Regulations (17 clauses: registration, credits, attendance, grading, examinations, summer term, progression, withdrawal, credit transfer, award of degree), code of conduct, fees, anti-ragging, etc.", "Main source of rules. Converted to handbook.md with one heading per section and clause."],
      ["SOP for Students (17 Aug 2026, PDF, 6 pages)", "Registration on Digii, late registration, add/drop, attendance, change of programme, summer term, fees, bonafide certificates, grievances, campus guidelines.", "Procedural rules; converted to sop.md (11 topics). Source of several rule conflicts with the handbook."],
      ["Semester Spread Structures Sept 2026 (XLSX, 10 sheets)", "For each batch 2022–2026: every course per semester (code, title, prerequisites, L-T-P, credits) and the credit-basket summary sheets.", "Course catalogue, prerequisites per batch, semester offerings, programme structure and credit baskets."],
      ["Minor Courses for B.Tech Students (XLSX, 7 sheets)", "Law, Design, Psychology, Economics, Finance, Marketing and Start-up minors per batch.", "Minor course lists, credits, semesters and prerequisites."],
    ],
    [2300, 3700, 3026],
  ),
);

// 3. Architecture
children.push(
  h1("3. System Architecture"),
  p(
    "The system has three parts, shown in Figure 1: (A) an offline data-preparation pipeline that runs once, (B) an online question-answering path that runs for every student message, and (C) an evaluation harness that drives the same online path with a fixed test set.",
  ),
  ...image("figures/pipeline.png", 600, 420, "Figure 1. End-to-end pipeline of the VU AI Academic Advisor"),
  h2("3.1 Offline data preparation (Lane A)"),
  numbered([["extract_data.py", { bold: true }], " reads the raw PDFs and spreadsheets and writes clean Markdown/JSON documents to data/processed/."]),
  numbered([["make_profiles.py", { bold: true }], " builds 10 synthetic students from the real semester spread (real courses and credits, invented grades)."]),
  numbered([["ingest.py", { bold: true }], " splits the processed documents into chunks, embeds them with the local all-MiniLM-L6-v2 model and stores 412 chunks in a persistent ChromaDB collection together with their citation metadata."]),
  h2("3.2 Online question answering (Lane B)"),
  numbered(["The Next.js chat UI sends the question, the chosen strategy, the optional student profile ID and the conversation history to the FastAPI endpoint POST /chat."], "steps2"),
  numbered(["advisor.py routes the request to one of the four strategies. For follow-up answers it merges the student's reply with their original question so retrieval still finds the right documents."], "steps2"),
  numbered(["For RAG strategies, retrieval.py fetches the most relevant chunks (semantic top-5 plus exact course lookups, prerequisite chain and the student's own semester lists)."], "steps2"),
  numbered(["prompts.py assembles the strategy-specific prompt; llm.py calls the fastest available model (Groq gpt-oss-120b, falling back automatically to gpt-oss-20b and Gemini)."], "steps2"),
  numbered(["The model's answer is constrained to a JSON schema and validated with Pydantic, then rendered in the UI with confidence, sources, clarifying question and warning banners."], "steps2"),
  h2("3.3 Evaluation (Lane C)"),
  p("run_eval.py sends each of the 25 test cases through all four strategies (plus three multi-turn follow-ups), labels every response against a verified expected outcome and writes results.csv, summary.md and summary.json."),
  h2("3.4 Technology choices"),
  table(
    ["Layer", "Technology", "Why"],
    [
      ["Embeddings", "sentence-transformers all-MiniLM-L6-v2 (CPU)", "Free, local, ~10 ms per query, good quality for short academic text."],
      ["Vector store", "ChromaDB (persistent, local)", "No server to run; metadata filters used for exact course/semester lookups."],
      ["LLM (primary)", "Groq openai/gpt-oss-120b, reasoning effort low", "Very low latency (~1–2 s) and strict JSON-schema structured output."],
      ["LLM (fallback)", "Groq gpt-oss-20b, Gemini flash-lite-latest, Gemini 3.5-flash-lite", "Keeps the advisor answering when a free-tier rate limit is reached."],
      ["API", "FastAPI + Pydantic", "Typed request/response models; the same schema validates LLM output."],
      ["Frontend", "Next.js 16, TypeScript, Tailwind CSS", "Single-page chat UI with live strategy switching for the demo."],
    ],
    [1700, 3300, 4026],
  ),
);

// 4. Data preparation
children.push(
  h1("4. Data Preparation"),
  h2("4.1 Student Handbook"),
  bullet("Text was extracted page by page with pypdf. Cover, contents and phone-number pages were skipped; the remaining pages were grouped into the handbook's 11 sections using a page map built by inspecting the PDF."),
  bullet("Section III (Academic Regulations) was split into its 17 numbered clauses (1. Academic Calendar … 17. Power to Revise) so every chunk can be cited as, for example, “Academic Regulations > 12. Progression (12.1)”."),
  bullet("The PDF contains invisible control characters (e.g. \\x03 between a clause number and its title) that broke heading detection; these are stripped before parsing."),
  bullet("Table 2 (letter grades and grade points, O = 10 … D = 4, F/FA = 0) is an image inside the PDF, so it was transcribed manually into the processed handbook and marked as such."),
  h2("4.2 SOP"),
  p("The SOP places topic labels in a side column, which PDF extraction interleaves with the text. The labels are removed and the text is split into its 11 topics using the first bullet of each topic as an anchor."),
  h2("4.3 Semester spread and programme structure"),
  bullet("Each batch sheet is a grid with a seven-column block per semester (code, title, prerequisite, L, T, P, credits). It is read by fixed column positions, carrying the credit basket (University Core, Foundation, Program Core, Honours, Specialisation/Electives, Minor/Open, Internship/Capstone) down the rows."),
  bullet("Placeholder slots such as “Minor/Open (1 Course)” or “Elective 1” are kept as TBA offerings."),
  bullet("A 135-course catalogue is built. Because curricula differ by batch, each course keeps its semester and prerequisites per batch rather than one merged value."),
  bullet("The Struct_* summary sheets (credit baskets per batch) are extracted separately, because they sometimes disagree with the semester-spread sheets."),
  h2("4.4 Minors"),
  p("All seven minor sheets were parsed into 166 rows (minor, batch, course, credits, semester, prerequisites). Rows marked TBA, TBD, “DON'T KNOW” or “New” are kept and labelled as not yet decided, so the advisor can say the information is not available."),
  h2("4.5 Current term (derived assumption)"),
  p("No academic calendar was supplied. The offering file is dated September 2026 and handbook clause 1.2 states that the odd semester runs July/August–December, so September 2026 is treated as the odd semester. The 2026 batch is therefore in Semester 1, 2025 in Semester 3, 2024 in Semester 5 and 2023 in Semester 7. This reasoning is stored in term_context.md and shown to the model, and it is reported as an assumption."),
  h2("4.6 Data-quality findings"),
  p("Parsing the real files revealed genuine gaps and contradictions. These became test cases for the “missing information” and “conflicting rules” requirements:"),
  table(
    ["Finding", "Evidence in the source", "Used as"],
    [
      ["Machine Learning prerequisites change by batch", "DATA301 needs DATA202 + MATH301 (2022–2024) but DATA206 + MATH301 (2025–2026)", "Conflict / clarify-batch test (X02)"],
      ["Undefined prerequisite code", "DATA206 appears only as a prerequisite; it exists nowhere else in the files", "Missing-information test (I01)"],
      ["University Core credits disagree", "2026 semester spread: 24 credits; Struct_2026_DS summary: 18 credits", "Conflict test (X03)"],
      ["Late-registration rules disagree", "SOP: “No late registration shall be permitted” vs Handbook 2.7: up to one calendar week with a late fee", "Conflict test (X01)"],
      ["Medical attendance approval differs", "Handbook 7.3: Vice-Chancellor approval; SOP: Faculty Advisor → Program Chair", "Conflict test (X04)"],
      ["Undefined minor prerequisites", "PSYC101, ECON202, FINA201 referenced but never defined", "Missing-information test (I04)"],
      ["Minor courses not decided", "Many 2025/2026 minor slots are TBA/TBD/“DON'T KNOW”", "Advisor must say the information is not available"],
    ],
    [2500, 4200, 2326],
  ),
);

// 5. Profiles
children.push(
  h1("5. Synthetic Student Profiles"),
  p(
    "The assignment provides no student data, so 10 anonymised profiles (IDs SYN-001 to SYN-010, no names or contact details) were generated. Each profile is built from its batch's real semester spread: the courses, codes, credits and semesters are genuine; only the grades and situations are invented. Earned credits and CGPA are computed with the handbook's own rules — clause 8.11 (credits are earned only with O, A+, A, B+, B, C, D or S) and clause 8.16 (CGPA is the credit-weighted grade-point average, with F and FA counting as zero).",
  ),
  table(
    ["ID", "Batch / sem", "Credits", "CGPA", "Failed", "Scenario tested"],
    profiles.map((s) => [
      s.student_id,
      s.batch ? `${s.batch} / S${s.current_semester}` : "unknown",
      s.completed_credits ?? "unknown",
      s.cgpa ?? "unknown",
      s.courses_failed && s.courses_failed.length ? s.courses_failed.map((f) => `${f.code} (${f.grade})`).join(", ") : s.courses_failed ? "none" : "unknown",
      s.scenario || "",
    ]),
    [900, 1150, 850, 750, 1700, 3676],
  ),
  gap(),
  p("The profiles cover the edge cases the brief asks for: a missing prerequisite (SYN-001, SYN-004), probation / progression risk (SYN-003, SYN-008), near graduation with distinction (SYN-002), a transfer-credit cap case (SYN-005), a medical attendance case (SYN-006), a deliberately incomplete record that forces clarification (SYN-007) and minor-programme cases (SYN-009, SYN-010)."),
  p("Each profile also stores a short “scenario” note for the UI. This note is deliberately removed from the prompt, because it would otherwise give the model the answer and inflate the evaluation."),
);

// 6. RAG
children.push(
  h1("6. Retrieval-Augmented Generation"),
  h2("6.1 Chunking"),
  bullet("Markdown documents are split on their ## and ### headings, so a chunk never mixes two clauses or two semesters."),
  bullet("Long clauses (e.g. clause 8, Grading) are split further at sub-clause numbers such as 8.13.3 into pieces of about 1,000 characters, because MiniLM only embeds about 256 tokens."),
  bullet("The course catalogue is split into one chunk per course. Each chunk lists its per-batch semester and prerequisites and flags undefined prerequisite codes."),
  bullet("Every chunk carries metadata used for citation: source_file (the original university document), section (the heading path plus sub-clause) and course_code."),
  table(
    ["Source", "Chunks"],
    [
      ["Student Handbook", "141"], ["SOP", "19"], ["Programme structure (semester spread)", "57"],
      ["Credit-structure summary sheets", "23"], ["Minors", "33"], ["Term context", "4"], ["Course catalogue records", "135"], ["Total", "412"],
    ],
    [6000, 3026],
  ),
  h2("6.2 Retrieval"),
  p("Retrieval combines four steps, so the model sees everything an advisor would check:"),
  numbered([["Semantic search: ", { bold: true }], "the question is embedded and the 5 most similar chunks are returned by cosine similarity."], "ret"),
  numbered([["Exact course lookup: ", { bold: true }], "course codes (e.g. DATA301) or exact course titles in the question are looked up directly by metadata, because embedding models match codes poorly."], "ret"),
  numbered([["Prerequisite chain (rag_structured): ", { bold: true }], "the catalogue records of every prerequisite of a retrieved course are added (Machine Learning → DATA202, MATH301 → MATH201)."], "ret"),
  numbered([["Student's own semesters (rag_structured): ", { bold: true }], "the course lists for the student's current and next semester in their batch are added."], "ret"),
  p("Retrieval runs locally in about 30–150 ms. The embedding model, database connection and API clients are loaded once at server start-up."),
  h2("6.3 Multi-turn conversations"),
  p("When the advisor asks a clarifying question, the student's next message is usually only an answer (“I'm in the 2024 batch”). The advisor therefore merges that reply with the student's original question to build the retrieval query, and the whole conversation is passed to the model inside <conversation_so_far> tags so the final answer uses both."),
);

// 7. Prompt engineering
children.push(
  h1("7. Prompt Engineering Strategies"),
  p("Four strategies were implemented in prompts.py. Each adds one layer of technique to the previous one, so the evaluation can attribute improvements to specific techniques (Figure 2)."),
  ...image("figures/strategies.png", 600, 195, "Figure 2. The four prompting strategies"),
  table(
    ["Strategy", "Techniques", "What it can and cannot do"],
    [
      ["1. baseline", "Raw question + “Answer this academic question from a university student”. No system prompt, no documents, free text.", "Represents simply asking an LLM. Answers from general knowledge; cannot cite sources or flag uncertainty."],
      ["2. structured", "Role/persona (VU academic advisor), explicit constraints (never invent codes/credits/dates, stay in scope, ask when facts are missing), XML delimiters around the question, JSON output with confidence and uncertainty flags.", "Makes the model more honest (it declines instead of guessing) but it still has no university data."],
      ["3. rag", "Structured + retrieved chunks inside <context>, each labelled with source_file and section; grounding rules: answer only from context, cite every chunk used, flag insufficient information and conflicting rules.", "Correct, sourced answers to general questions; cannot judge an individual student's eligibility."],
      ["4. rag_structured", "RAG + the student's structured profile, the current-term context, prerequisite and semester lookups, and a five-step eligibility procedure (identify course → check prerequisites → check offering term → check standing → decide), with clarifying questions when data is missing.", "Personalised eligibility decisions with reasons and sources; asks for missing facts."],
    ],
    [1500, 4300, 3226],
  ),
  h2("7.1 Excerpt: constraints and delimiters"),
  ...code([
    "<constraints>",
    "- Never invent course codes, credit numbers, CGPA thresholds, deadlines or rules. If you do not",
    "  know a university-specific fact with certainty, say so and set insufficient_information to true.",
    "- If the question depends on facts about the student you do not have, do NOT assume: set",
    "  needs_clarification to true and ask ONE specific clarifying_question.",
    "</constraints>",
    "<student_question> ... </student_question>",
  ]),
  h2("7.2 Excerpt: eligibility procedure (rag_structured)"),
  ...code([
    "1. Identify the course(s) or rule involved and the student's batch (curriculum year).",
    "2. Prerequisites: mark each as passed, failed (F/FA) or missing. A course with F or FA is NOT passed.",
    "3. Offering: find the course's semester for the batch; use <term_context> to decide if it runs.",
    "4. Standing: check CGPA against progression / degree thresholds and credits against requirements.",
    "5. Decide: eligible / not eligible / cannot determine, with reasons and documented options only.",
  ]),
  h2("7.3 Structured output"),
  p("Strategies 2–4 must return this JSON, enforced by a strict JSON schema on the model and validated with Pydantic. The baseline returns free text, which is wrapped with fixed default flags."),
  table(
    ["Field", "Meaning"],
    [
      ["answer", "Reply to the student (≈150 words max)"],
      ["confidence", "high / medium / low"],
      ["grounded", "True only if every claim comes from the provided documents"],
      ["sources[]", "{source_file, section} for every chunk used"],
      ["needs_clarification, clarifying_question", "Set when the advisor must ask before answering"],
      ["insufficient_information", "The documents cannot answer reliably"],
      ["conflict_detected", "Sources disagree (added to the brief's schema so conflict handling can be measured)"],
    ],
    [3300, 5726],
  ),
);

// 8. LLM + latency
children.push(
  h1("8. LLM Layer and Latency Design"),
  p("Low latency was a design goal, so the LLM layer uses Groq (very fast inference) with automatic fallbacks:"),
  table(
    ["Order", "Model", "Role"],
    [
      ["1", "Groq openai/gpt-oss-120b (reasoning effort low)", "Primary: best quality at ~1–2 s"],
      ["2", "Groq openai/gpt-oss-20b", "Same API, separate rate-limit budget"],
      ["3", "Gemini gemini-flash-lite-latest", "Fallback (~1.5 s)"],
      ["4", "Gemini gemini-3.5-flash-lite", "Last resort"],
    ],
    [900, 4600, 3526],
  ),
  gap(),
  bullet([["No SDK retries: ", { bold: true }], "a failing model is skipped immediately rather than retried."]),
  bullet([["Rate-limit cool-down: ", { bold: true }], "Groq's free tier allows 8,000 tokens per minute per model. When a model returns HTTP 429 it is marked unavailable for its retry-after period and skipped instantly on later requests."]),
  bullet([["Compact prompts: ", { bold: true }], "the profile is rendered as one line per course instead of indented JSON and duplicate labels are removed (the rag_structured prompt shrank from about 5,700 to about 3,700 tokens); max_tokens is 1,000 because Groq reserves it against the per-minute budget."]),
  bullet([["Strict JSON schema: ", { bold: true }], "the model can only return valid JSON, so there is no re-parsing or retrying."]),
  bullet([["Warm start and thread safety: ", { bold: true }], "the embedding model, ChromaDB and API clients are loaded at start-up; embeddings run on CPU behind a lock (see Section 12)."]),
  h2("8.1 Measured latency (live /chat calls)"),
  table(
    ["Query type", "Total time", "Retrieval time", "Answered by"],
    [
      ["Normal question (CGPA for Year 2)", "1.69 s", "157 ms (first call)", "gpt-oss-120b"],
      ["Ambiguous question (ML next semester)", "1.78 s", "35 ms", "gpt-oss-120b"],
      ["Missing information (add/drop date)", "1.18 s", "34 ms", "gpt-oss-120b"],
      ["Eligibility with profile SYN-001", "1.35 s", "41 ms", "gpt-oss-20b (fallback after 429)"],
    ],
    [3600, 1500, 1900, 2026],
  ),
);

// 9. Frontend
children.push(
  h1("9. User Interface"),
  p("The Next.js chat application (http://localhost:3000) is designed for a clear live demonstration:"),
  bullet("Vidyashilp University branding, backend status indicator and example questions."),
  bullet("Student-profile selector (anonymous or SYN-001…SYN-010, showing batch, semester and CGPA) and a strategy selector, so strategies can be switched live during the presentation."),
  bullet("Each answer shows a confidence badge, a grounded/not-grounded badge, the strategy, total and retrieval time, the model that answered, and an expandable Sources list with document and section."),
  bullet("A clarifying question is highlighted in an indigo box; the input field is focused and its placeholder changes to “Answer the advisor's question…”."),
  bullet("An amber banner appears when information is insufficient and an orange banner when the documents contain conflicting rules."),
);

// 10. Evaluation methodology
const catCount = {};
cases.forEach((c) => (catCount[c.category] = (catCount[c.category] || 0) + 1));
children.push(
  h1("10. Evaluation Methodology"),
  p("A fixed test set of 25 academic queries was written, each with an expected outcome verified by hand against the processed documents (clause numbers, prerequisite codes, profile CGPAs). Every case is run through all four strategies on the same pinned model (Groq gpt-oss-120b), so differences come from the strategy and not the model."),
  table(
    ["Category", "Cases", "What is tested"],
    [
      ["factual", catCount.factual, "Straight facts: attendance %, progression CGPA, degree credits/CGPA, prerequisites, summer-term credit cap"],
      ["eligibility", catCount.eligibility, "Decisions that need the student profile (prerequisites, progression, distinction, transfer-credit cap)"],
      ["unavailable_course", catCount.unavailable_course, "Course is valid but not offered in the current (odd) term"],
      ["ambiguous", catCount.ambiguous, "Advisor must ask a clarifying question; each has a scripted follow-up answer"],
      ["insufficient_info", catCount.insufficient_info, "Answer is not in the data (undefined codes, exact dates, fee amounts, future offerings)"],
      ["conflicting_rules", catCount.conflicting_rules, "Sources disagree; advisor must present both rules"],
    ],
    [2000, 800, 6226],
  ),
  h2("10.1 Labelling rules"),
  p("Each response receives one of four labels using rule-based checks: the structured flags when present, OR keyword phrases in the answer text (so the baseline, which has no flags, is judged on what it says):"),
  table(
    ["Case type", "correct", "partially_correct", "unsupported_hallucination", "incorrect"],
    [
      ["Answer (factual / eligibility / unavailable)", "All required facts present and the right eligibility decision", "Some facts, or the right decision with facts missing", "Asserted an answer containing none of the required facts", "Wrong eligibility decision, or abstained when the answer was available"],
      ["Clarify / insufficient", "Expected behaviour detected", "The other uncertainty behaviour (asked instead of saying insufficient, or vice versa)", "Answered confidently anyway", "—"],
      ["Conflict", "Conflict (or accepted alternative) flagged and required facts present", "Only one of the two", "—", "Presented a single rule as the only rule"],
    ],
    [1900, 1850, 1850, 1700, 1726],
  ),
  h2("10.2 Metrics"),
  bullet([["Accuracy: ", { bold: true }], "share of cases labelled correct (also reported with partial answers counted as 0.5)."]),
  bullet([["Hallucination rate: ", { bold: true }], "share labelled unsupported_hallucination."]),
  bullet([["Correct eligibility decisions: ", { bold: true }], "eligible / not-eligible read from the answer matches the verified decision."]),
  bullet([["Source/evidence correctness: ", { bold: true }], "at least one cited source matches the expected document section (strategies without retrieval cite nothing and score 0)."]),
  bullet([["Missing-info, conflicting-rules and clarification handling: ", { bold: true }], "share of those categories answered correctly."]),
  bullet([["Final answer after follow-up: ", { bold: true }], "for the 3 ambiguous cases, the student answers the clarifying question and the second response is scored — this measures whether asking the right question improves the final recommendation."]),
  bullet([["Response time: ", { bold: true }], "average wall-clock seconds per call, including retrieval."]),
);

// 11. Results
children.push(h1("11. Results"));
if (summary) {
  children.push(
    p(`All strategies were run on the same model (${(summary.providers || []).join(", ")}).`),
    table(
      ["Metric", "Baseline", "Structured", "RAG", "RAG + Profile"],
      summary.metrics.map((m) => [m.metric, m.baseline, m.structured, m.rag, m.rag_structured]),
      [3226, 1450, 1450, 1450, 1450],
    ),
  );
  if (fs.existsSync(path.join(__dirname, "figures/results.png")))
    children.push(...image("figures/results.png", 600, 300, "Figure 3. Key metrics across the four development stages"));
  children.push(
    h2("11.1 Correct answers by category"),
    table(
      ["Category (n)", "Baseline", "Structured", "RAG", "RAG + Profile"],
      summary.categories.map((c) => [`${c.category} (${c.n})`, c.baseline, c.structured, c.rag, c.rag_structured]),
      [3226, 1450, 1450, 1450, 1450],
    ),
    h2("11.2 Label distribution"),
    table(
      ["Strategy", "Correct", "Partial", "Hallucinated", "Incorrect"],
      Object.entries(summary.labels).map(([s, l]) => [s, l.correct, l.partially_correct, l.unsupported_hallucination, l.incorrect]),
      [3226, 1450, 1450, 1450, 1450],
    ),
  );
} else {
  children.push(p("[Evaluation still running — rebuild this document after eval/run_eval.py finishes.]"));
}
children.push(
  h2("11.3 Example responses (rag_structured, raw JSON excerpts)"),
  p("Normal question — “What minimum CGPA do I need to be promoted to the second year?”"),
  ...code([
    '"answer": "...you must have a minimum CGPA of 4.00 at the end of the relevant semester.",',
    '"confidence": "high", "grounded": true,',
    '"sources": [{"source_file": "Student Handbook Aug 2026.pdf",',
    '             "section": "Section III - Academic Regulations > 12. Progression (12.1)"}]',
  ]),
  p("Ambiguous question — “Can I take Machine Learning next semester?” (no profile)"),
  ...code([
    '"needs_clarification": true,',
    '"clarifying_question": "Which batch are you in, and have you passed the prerequisite courses',
    '   listed for your batch (e.g., DATA202 or DATA206 and MATH301)?"',
  ]),
  p("Missing information — “What is the exact date of the add/drop deadline this semester?”"),
  ...code([
    '"answer": "...the exact add/drop deadline date for the current semester is not provided in the',
    '   available documents.", "insufficient_information": true,',
    '"sources": [{"section": "Section III - Academic Regulations > 2. Registration (2.15)"}]',
  ]),
  p("Eligibility with profile SYN-001 — “Can I register for Machine Learning this semester?”"),
  ...code([
    '"answer": "You cannot register for DATA301 Machine Learning ... For the 2024 batch, DATA301',
    '   requires both DATA202 and MATH301. You passed DATA202 but failed MATH301 ..."',
    '"sources": [{"section": "Course DATA301 Machine Learning"}, {"section": "2024 Batch - Semester 5"}]',
  ]),
);

// 12. Engineering issues
children.push(
  h1("12. Problems Encountered and How They Were Solved"),
  table(
    ["Problem", "Cause", "Solution"],
    [
      ["Only 2 of 17 handbook clauses detected", "Invisible PDF control characters (\\x03) inside headings", "Strip control characters before parsing"],
      ["2026 batch missing its Program Core basket", "Basket label row lacked the “B3” marker used for detection", "Detect baskets by label + numeric credit value"],
      ["Catalogue hid batch differences", "One merged prerequisite value per course", "Store semester and prerequisites per batch; flag differences and undefined codes"],
      ["Minor prerequisites contained the semester number", "Off-by-one column index", "Fixed index; verified against the sheet"],
      ["Evaluation leakage", "Profile “scenario” note (e.g. “needs MATH301”) was sent to the model", "Removed from the prompt; UI-only field"],
      ["HTTP 429 on Groq", "Free tier: 8,000 tokens/min per model; prompts were ~5,700 tokens", "Compact prompts, lower max_tokens, fallback chain with cool-down"],
      ["Gemini 503 / retired models", "gemini-2.5 models unavailable; flash-latest overloaded", "Use flash-lite models as fallbacks"],
      ["Evaluation froze twice", "Two threads ran the embedding model on the Apple GPU (MPS) at once and deadlocked", "Run embeddings on CPU behind a lock; 20 concurrent retrievals now finish in 0.17 s"],
      ["Results lost when the run was killed", "Results were saved only at the end", "Save each result immediately; resume skips completed cases"],
    ],
    [2600, 3100, 3326],
  ),
);

// 13. Limitations
children.push(
  h1("13. Limitations"),
  bullet("The current term is derived from the file date and handbook clause 1.2; no academic calendar was provided, so exact dates cannot be answered."),
  bullet("PDF and spreadsheet extraction is heuristic (regex headings, fixed column positions); the grade table was transcribed by hand."),
  bullet("Evaluation labels are rule-based keyword and flag matches. Paraphrased answers can be mis-labelled, so labels should be spot-checked by hand."),
  bullet("The baseline cannot self-report confidence, sources or flags, so it is judged only on its answer text."),
  bullet("Student profiles are synthetic; real transcripts may contain cases not covered."),
  bullet("Free-tier rate limits mean that under load some answers come from a fallback model; the provider field shows which."),
  bullet("The advisor does not replace the Faculty Advisor or Program Chair; for conflicting rules it tells the student whom to confirm with."),
);

// 14. Running
children.push(
  h1("14. How to Run the System"),
  h3("Backend"),
  ...code([
    "cd backend",
    "python3 -m venv .venv && source .venv/bin/activate",
    "pip install -r requirements.txt",
    "cp .env.example .env        # add GROQ_API_KEY and GEMINI_API_KEY",
    "python extract_data.py      # raw files -> data/processed",
    "python make_profiles.py     # synthetic students",
    "python ingest.py            # chunk + embed + ChromaDB",
    "uvicorn main:app --port 8000",
  ]),
  h3("Frontend"),
  ...code(["cd frontend", "cp .env.local.example .env.local", "npm install", "npm run dev      # http://localhost:3000"]),
  h3("Evaluation"),
  ...code(["LLM_PIN=groq:openai/gpt-oss-120b backend/.venv/bin/python eval/run_eval.py", "backend/.venv/bin/python eval/run_eval.py --rescore   # recompute metrics only"]),
  h2("14.1 Repository structure"),
  table(
    ["Path", "Contents"],
    [
      ["data/raw/", "The four university files"],
      ["data/processed/", "handbook.md, sop.md, programme_structure.md, structure_summary.md, minors.md, term_context.md, course_catalogue.json, semester_offerings.json, minors.json"],
      ["data/synthetic/", "student_profiles.json"],
      ["backend/", "config.py, extract_data.py, make_profiles.py, ingest.py, retrieval.py, prompts.py, llm.py, advisor.py, schemas.py, main.py"],
      ["frontend/", "Next.js app (app/page.tsx), logo in public/"],
      ["eval/", "test_cases.json, run_eval.py, raw_results.jsonl, results.csv, summary.md, summary.json"],
      ["docs/", "This document, its generator and figures"],
    ],
    [2200, 6826],
  ),
);

// Appendix
children.push(
  h1("Appendix A. Test Cases"),
  table(
    ["ID", "Category", "Query", "Expected outcome"],
    cases.map((c) => [c.id + (c.student_id ? `\n${c.student_id}` : ""), c.category, c.query, c.expected_outcome]),
    [800, 1400, 3200, 3626],
  ),
);

// ---------- document ----------
const doc = new Document({
  creator: "VU AI Academic Advisor project",
  title: "VU AI Academic Advisor — Technical Documentation",
  styles: {
    default: { document: { run: { font: FONT, size: 21 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, font: FONT, color: NAVY }, paragraph: { spacing: { before: 240, after: 200 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: FONT, color: NAVY }, paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 22, bold: true, font: FONT, color: "333333" }, paragraph: { spacing: { before: 160, after: 80 }, outlineLevel: 2 } },
    ],
  },
  numbering: {
    config: [
      { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      ...["steps", "steps2", "ret"].map((reference) => ({
        reference,
        levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }],
      })),
    ],
  },
  sections: [
    {
      properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
      headers: {
        default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [t("VU AI Academic Advisor — DATA308 Assignment #1", { size: 16, color: "888888" })] })] }),
      },
      footers: {
        default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [t("Page ", { size: 16, color: "888888" }), new TextRun({ children: [PageNumber.CURRENT], size: 16, color: "888888", font: FONT })] })] }),
      },
      children,
    },
  ],
});

Packer.toBuffer(doc).then((buf) => {
  const out = path.join(__dirname, "VU_AI_Academic_Advisor_Documentation.docx");
  fs.writeFileSync(out, buf);
  console.log("wrote", out);
});
