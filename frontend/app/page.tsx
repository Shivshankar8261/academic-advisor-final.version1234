"use client";

import Image from "next/image";
import { memo, useCallback, useEffect, useRef, useState } from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type Strategy = "baseline" | "structured" | "rag" | "rag_structured" | "agentic";

const STRATEGIES: { value: Strategy; label: string }[] = [
  { value: "rag_structured", label: "RAG + Student Profile (final)" },
  { value: "agentic", label: "Agentic (6-step pipeline)" },
  { value: "rag", label: "RAG" },
  { value: "structured", label: "Structured prompt (no docs)" },
  { value: "baseline", label: "Baseline LLM" },
];

type Source = { source_file: string; section: string };

type AdvisorReply = {
  answer: string;
  confidence: "high" | "medium" | "low";
  grounded: boolean;
  sources: Source[];
  needs_clarification: boolean;
  clarifying_question: string | null;
  insufficient_information: boolean;
  conflict_detected: boolean;
  strategy: Strategy;
  provider: string;
  response_time_s: number;
  retrieval_time_s: number;
  retrieved: { source_file: string; section: string; score: number }[];
  usage: { prompt_tokens: number; completion_tokens: number; total_tokens: number; estimated_cost_usd: number | null };
  agent_trace: { agent: string; summary: string; ms: number; tokens: number }[];
  attachments: { type: string; batch: string }[];
  guard: string | null;
  privacy_notice: string | null;
  human_review: string | null;
};

type RecordCourse = { code: string; grade: string };
type SessionRecord = { batch: string; current_semester: number; courses: RecordCourse[] };
type TranscriptDraft = {
  batch: string | null;
  current_semester: number | null;
  courses: RecordCourse[];
  rejected: string[];
  unreadable_rows: number;
};

const GRADES = ["O", "A+", "A", "B+", "B", "C", "D", "F", "FA", "S", "U", "I"];
const SILENT_MIC_RMS = 0.0005; // quieter than any real microphone's noise floor: only a dead input stays below this
const BATCHES = ["2022", "2023", "2024", "2025", "2026"];

type Message =
  | { role: "user"; content: string }
  | { role: "assistant"; reply: AdvisorReply }
  | { role: "error"; content: string };

type StudentSummary = {
  student_id: string;
  batch: string | null;
  current_semester: number | null;
  cgpa: number | null;
  completed_credits: number | null;
  scenario: string;
};

const EXAMPLES: { icon: string; tag: string; q: string }[] = [
  { icon: "📘", tag: "Prerequisites", q: "Can I take Machine Learning next semester?" },
  { icon: "🧮", tag: "Course lists", q: "How many courses are in the Foundation basket for the 2022 batch?" },
  { icon: "📝", tag: "Registration", q: "I missed registration because of a family function. Can I still register late?" },
  { icon: "🗺️", tag: "Infographic", q: "Show me an infographic of the 2024 batch course structure" },
];

function formatCost(cost: number | null): string {
  if (cost === null) return "free tier";
  return cost < 0.0001 ? "<$0.0001" : `$${cost.toFixed(4)}`;
}

// Shrinks the photo before upload (Vercel caps request bodies at 4.5 MB); re-encoding also drops EXIF data.
async function prepareImage(file: File): Promise<Blob> {
  const bitmap = await createImageBitmap(file);
  const scale = Math.min(1, 1800 / Math.max(bitmap.width, bitmap.height));
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(bitmap.width * scale);
  canvas.height = Math.round(bitmap.height * scale);
  canvas.getContext("2d")?.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  return new Promise((resolve, reject) =>
    canvas.toBlob((b) => (b ? resolve(b) : reject(new Error("Could not read that image."))), "image/jpeg", 0.9),
  );
}

const CONFIDENCE_STYLE: Record<string, { pill: string; dot: string }> = {
  high: { pill: "bg-emerald-50 text-emerald-700 ring-emerald-200", dot: "bg-emerald-500" },
  medium: { pill: "bg-amber-50 text-amber-700 ring-amber-200", dot: "bg-amber-500" },
  low: { pill: "bg-rose-50 text-rose-700 ring-rose-200", dot: "bg-rose-500" },
};

// LLM answers often contain Markdown (bold, lists, tables, headings) — render it instead of showing raw "**".
const MD: Components = {
  p: (props) => <p className="my-2 first:mt-0 last:mb-0" {...props} />,
  h1: (props) => <h3 className="mb-2 mt-4 text-base font-semibold text-brand-900 first:mt-0" {...props} />,
  h2: (props) => <h3 className="mb-2 mt-4 text-base font-semibold text-brand-900 first:mt-0" {...props} />,
  h3: (props) => <h4 className="mb-1.5 mt-3 font-semibold text-brand-900 first:mt-0" {...props} />,
  h4: (props) => <h4 className="mb-1.5 mt-3 font-semibold text-slate-800 first:mt-0" {...props} />,
  strong: (props) => <strong className="font-semibold text-slate-900" {...props} />,
  ul: (props) => <ul className="my-2 list-disc space-y-1 pl-5 marker:text-brand-500" {...props} />,
  ol: (props) => <ol className="my-2 list-decimal space-y-1 pl-5 marker:text-brand-600" {...props} />,
  a: (props) => <a className="text-brand-600 underline underline-offset-2" target="_blank" rel="noreferrer" {...props} />,
  hr: () => <hr className="my-3 border-slate-200" />,
  blockquote: (props) => <blockquote className="my-2 border-l-4 border-brand-100 pl-3 text-slate-600" {...props} />,
  code: (props) => <code className="rounded bg-slate-100 px-1 py-0.5 font-mono text-[13px] text-brand-700" {...props} />,
  table: (props) => (
    <div className="my-3 overflow-x-auto rounded-xl ring-1 ring-slate-200">
      <table className="w-full border-collapse text-left text-sm [&_td:nth-child(5)]:whitespace-nowrap" {...props} />
    </div>
  ),
  thead: (props) => <thead className="bg-brand-50 text-brand-900" {...props} />,
  th: (props) => <th className="px-3 py-2 font-semibold" {...props} />,
  td: (props) => <td className="border-t border-slate-100 px-3 py-2 align-top" {...props} />,
};

// FastAPI returns {detail: string} or {detail: [{msg}]} (validation); fall back to the status text.
async function errorText(res: Response): Promise<string> {
  try {
    const body = await res.json();
    if (typeof body?.detail === "string") return body.detail;
    if (Array.isArray(body?.detail)) return body.detail.map((d: { msg?: string }) => d.msg ?? String(d)).join("; ");
  } catch {
    /* non-JSON error body */
  }
  return `${res.status} ${res.statusText || "request failed"}`;
}

function AdvisorAvatar() {
  return (
    <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-gradient-to-br from-brand-500 to-brand-900 text-white shadow-md shadow-brand-500/30">
      <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth={1.8}>
        <path d="M22 10 12 5 2 10l10 5 10-5Z" strokeLinejoin="round" />
        <path d="M6 12v5c3 2 9 2 12 0v-5" strokeLinejoin="round" />
      </svg>
    </div>
  );
}

// Tables are on screen, and reading them cell by cell is useless, so only the sentences are spoken.
function speakableSentences(text: string): string[] {
  const plain = text
    .split("\n")
    .filter((l) => !l.trim().startsWith("|") && !/^-{3,}$/.test(l.trim()))
    .join(" ")
    .replace(/[*_`#>]/g, "")
    .replace(/\s+/g, " ")
    .trim();
  return plain.match(/[^.!?]+[.!?]*\s*/g) ?? [plain];
}

function ReadAloud({ text }: { text: string }) {
  const [state, setState] = useState<"idle" | "loading" | "playing">("idle");
  const audioRef = useRef<HTMLAudioElement | null>(null);

  // Stop speaking if this message leaves the screen (New chat).
  useEffect(() => () => window.speechSynthesis?.cancel(), []);

  // The cloud voice takes up to ~30 s on a long answer, so it is only the fallback for browsers with no voices.
  async function cloudVoice() {
    setState("loading");
    try {
      const res = await fetch(`${API}/speech`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      if (!res.ok) throw new Error(await errorText(res));
      const audio = new Audio(URL.createObjectURL(await res.blob()));
      audioRef.current = audio;
      audio.onended = () => setState("idle");
      setState("playing");
      await audio.play();
    } catch {
      setState("idle");
    }
  }

  function toggle() {
    if (state === "playing") {
      audioRef.current?.pause();
      window.speechSynthesis?.cancel();
      setState("idle");
      return;
    }
    const synth = window.speechSynthesis;
    const voices = synth?.getVoices() ?? [];
    if (!synth) {
      cloudVoice();
      return;
    }
    const voice =
      voices.find((v) => v.lang === "en-IN" && v.localService) ??
      voices.find((v) => v.lang.startsWith("en") && v.localService) ??
      voices.find((v) => v.lang.startsWith("en"));
    synth.cancel();
    setState("playing");
    // One utterance per sentence: speech starts at once, and Chrome cuts off single utterances longer than ~15 s.
    const sentences = speakableSentences(text);
    sentences.forEach((sentence, i) => {
      const utterance = new SpeechSynthesisUtterance(sentence);
      if (voice) utterance.voice = voice;
      if (i === sentences.length - 1) utterance.onend = () => setState("idle");
      utterance.onerror = (e) => {
        if (e.error === "canceled" || e.error === "interrupted") return;
        synth.cancel();
        cloudVoice();
      };
      synth.speak(utterance);
    });
  }

  return (
    <button
      type="button"
      onClick={toggle}
      disabled={state === "loading"}
      className="inline-flex items-center gap-1 rounded-lg px-2 py-1 font-medium text-brand-700 hover:bg-brand-50 disabled:opacity-50"
    >
      {state === "loading" ? "⏳ Preparing audio…" : state === "playing" ? "⏹ Stop" : "🔊 Listen"}
    </button>
  );
}

function CopyForAdvisor({ reply, question }: { reply: AdvisorReply; question: string }) {
  const [copied, setCopied] = useState(false);
  async function copy() {
    const sources = reply.sources.map((s) => `- ${s.source_file}: ${s.section}`).join("\n") || "- none cited";
    await navigator.clipboard.writeText(
      `Question: ${question}\n\nAI advisor's answer (please check):\n${reply.answer}\n\nSources:\n${sources}`,
    );
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }
  return (
    <button
      type="button"
      onClick={copy}
      className="inline-flex items-center gap-1 rounded-lg px-2 py-1 font-medium text-brand-700 hover:bg-brand-50"
    >
      {copied ? "✓ Copied" : "📋 Copy for my advisor"}
    </button>
  );
}

const AssistantMessage = memo(function AssistantMessage({ reply, question }: { reply: AdvisorReply; question: string }) {
  const conf = CONFIDENCE_STYLE[reply.confidence] ?? CONFIDENCE_STYLE.low;
  const smallTalk = reply.provider === "small-talk"; // greetings: no badges or sources
  return (
    <div className="flex max-w-full gap-3 sm:max-w-[88%]">
      <AdvisorAvatar />
      <div className="min-w-0 flex-1 rounded-2xl rounded-tl-md border border-white/70 bg-white p-5 shadow-md shadow-slate-200/60 ring-1 ring-slate-200/70">
        {reply.insufficient_information && !reply.needs_clarification && (
          <div className="mb-3 flex gap-2 rounded-xl border border-amber-200 bg-gradient-to-r from-amber-50 to-yellow-50 px-3 py-2.5 text-sm font-medium text-amber-900">
            <span>⚠️</span>
            <span>I don&apos;t have enough information in the university documents to answer this reliably.</span>
          </div>
        )}
        {reply.conflict_detected && (
          <div className="mb-3 flex gap-2 rounded-xl border border-orange-200 bg-gradient-to-r from-orange-50 to-rose-50 px-3 py-2.5 text-sm font-medium text-orange-900">
            <span>⚖️</span>
            <span>
              The documents contain conflicting rules on this — please confirm with your Faculty Advisor / Program
              Chair.
            </span>
          </div>
        )}

        {reply.human_review && (
          <div className="mb-3 flex gap-2 rounded-xl border border-violet-200 bg-violet-50 px-3 py-2.5 text-sm font-medium text-violet-900">
            <span>🧑‍🏫</span>
            <span>{reply.human_review}</span>
          </div>
        )}
        {reply.privacy_notice && (
          <div className="mb-3 flex gap-2 rounded-xl border border-sky-200 bg-sky-50 px-3 py-2.5 text-sm font-medium text-sky-900">
            <span>🔒</span>
            <span>{reply.privacy_notice}</span>
          </div>
        )}

        <div className="break-words text-[15px] leading-relaxed text-slate-800">
          <ReactMarkdown remarkPlugins={[remarkGfm]} components={MD}>
            {reply.answer}
          </ReactMarkdown>
        </div>

        {reply.attachments.map((a) => (
          <figure key={a.batch} className="mt-3 overflow-hidden rounded-xl ring-1 ring-slate-200">
            {/* eslint-disable-next-line @next/next/no-img-element -- SVG served by the API, sized by its viewBox */}
            <img
              src={`${API}/infographic/${a.batch}`}
              alt={`Course structure infographic for the ${a.batch} batch: credits by basket and courses by semester`}
              className="w-full"
            />
            <figcaption className="flex items-center justify-between bg-slate-50 px-3 py-1.5 text-xs text-slate-500">
              <span>Built from the semester spread. Not an official University document.</span>
              <a className="font-medium text-brand-700 underline" href={`${API}/infographic/${a.batch}`} target="_blank" rel="noreferrer">
                Open full size
              </a>
            </figcaption>
          </figure>
        ))}

        {reply.needs_clarification && reply.clarifying_question && (
          <div className="mt-4 rounded-xl border border-brand-100 bg-gradient-to-r from-brand-50 to-violet-50 px-4 py-3 text-[15px] text-brand-900">
            <div className="mb-1 text-xs font-semibold uppercase tracking-wide text-brand-600">💬 Question for you</div>
            {reply.clarifying_question}
          </div>
        )}

        {!smallTalk && (
        <>
        <div className="mt-4 flex flex-wrap items-center gap-1.5 text-xs">
          <span className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 font-medium ring-1 ${conf.pill}`}>
            <span className={`h-1.5 w-1.5 rounded-full ${conf.dot}`} />
            {reply.confidence} confidence
          </span>
          <span
            className={`inline-flex items-center gap-1 rounded-full px-2.5 py-1 font-medium ring-1 ${
              reply.grounded ? "bg-sky-50 text-sky-700 ring-sky-200" : "bg-slate-50 text-slate-500 ring-slate-200"
            }`}
          >
            {reply.grounded ? "✓ grounded in documents" : "not grounded"}
          </span>
          <span className="inline-flex items-center gap-1 rounded-full bg-slate-50 px-2.5 py-1 text-slate-600 ring-1 ring-slate-200">
            ⚡ {reply.response_time_s.toFixed(2)}s
            {reply.retrieval_time_s > 0 && (
              <span className="text-slate-400"> · retrieval {(reply.retrieval_time_s * 1000).toFixed(0)} ms</span>
            )}
          </span>
          <span
            title={`${reply.usage.prompt_tokens} input + ${reply.usage.completion_tokens} output tokens`}
            className="inline-flex items-center gap-1 rounded-full bg-slate-50 px-2.5 py-1 text-slate-600 ring-1 ring-slate-200"
          >
            🔢 {reply.usage.total_tokens.toLocaleString()} tokens
            <span className="text-slate-400">
              {" "}
              · {reply.usage.total_tokens ? formatCost(reply.usage.estimated_cost_usd) : "no model used"}
            </span>
          </span>
          <span className="rounded-full bg-slate-50 px-2.5 py-1 font-mono text-[11px] text-slate-500 ring-1 ring-slate-200">
            {reply.strategy} · {reply.provider}
          </span>
        </div>

        <div className="mt-2 flex flex-wrap items-center gap-1 text-sm">
          <ReadAloud text={reply.answer} />
          <CopyForAdvisor reply={reply} question={question} />
        </div>

        {reply.agent_trace.length > 0 && (
          <details className="group mt-2 text-sm">
            <summary className="inline-flex cursor-pointer select-none list-none items-center gap-1.5 rounded-lg px-2 py-1 font-medium text-brand-700 hover:bg-brand-50 [&::-webkit-details-marker]:hidden">
              🤖 How the agents worked ({reply.agent_trace.length} steps)
            </summary>
            <ol className="mt-2 grid gap-1.5">
              {reply.agent_trace.map((t, i) => (
                <li key={i} className="rounded-lg border border-slate-100 bg-slate-50/80 px-3 py-2">
                  <span className="font-medium text-slate-800">
                    {i + 1}. {t.agent}
                  </span>
                  <span className="text-xs text-slate-400">
                    {" "}
                    · {t.ms} ms{t.tokens ? ` · ${t.tokens.toLocaleString()} tokens` : ""}
                  </span>
                  <div className="text-slate-600">{t.summary}</div>
                </li>
              ))}
            </ol>
          </details>
        )}

        <details className="group mt-2 text-sm">
          <summary className="inline-flex cursor-pointer select-none list-none items-center gap-1.5 rounded-lg px-2 py-1 font-medium text-brand-700 hover:bg-brand-50 [&::-webkit-details-marker]:hidden">
            <svg
              viewBox="0 0 20 20"
              className="h-3.5 w-3.5 transition-transform group-open:rotate-90"
              fill="currentColor"
            >
              <path d="M7 5l6 5-6 5V5z" />
            </svg>
            📚 Sources ({reply.sources.length} cited
            {reply.retrieved.length ? `, ${reply.retrieved.length} retrieved` : ""})
          </summary>
          {reply.sources.length === 0 ? (
            <p className="mt-2 pl-2 text-slate-500">No sources cited.</p>
          ) : (
            <ul className="mt-2 grid gap-1.5">
              {reply.sources.map((s, i) => (
                <li
                  key={i}
                  className="flex items-start gap-2 rounded-lg border border-slate-100 bg-slate-50/80 px-3 py-2"
                >
                  <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-md bg-brand-100 text-[11px] font-semibold text-brand-700">
                    {i + 1}
                  </span>
                  <span>
                    <span className="font-medium text-slate-800">{s.source_file}</span>
                    <span className="text-slate-500"> — {s.section}</span>
                  </span>
                </li>
              ))}
            </ul>
          )}
        </details>
        </>
        )}
      </div>
    </div>
  );
});

function TypingIndicator() {
  return (
    <div className="flex animate-fade-up items-center gap-3 self-start">
      <AdvisorAvatar />
      <div className="flex items-center gap-3 rounded-2xl rounded-tl-md bg-white/90 px-4 py-3 shadow-md ring-1 ring-slate-200/70">
        <div className="flex gap-1">
          {[0, 150, 300].map((d) => (
            <span
              key={d}
              className="h-2 w-2 animate-typing rounded-full bg-brand-500"
              style={{ animationDelay: `${d}ms` }}
            />
          ))}
        </div>
        <span className="text-sm text-slate-500">Checking the university documents…</span>
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string | number | null }) {
  return (
    <div className="rounded-lg bg-white/80 px-3 py-1.5 ring-1 ring-brand-100">
      <div className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">{label}</div>
      <div className="text-sm font-semibold text-brand-900">{value ?? "—"}</div>
    </div>
  );
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [strategy, setStrategy] = useState<Strategy>("rag_structured");
  const [studentId, setStudentId] = useState("");
  const [students, setStudents] = useState<StudentSummary[]>([]);
  const [loading, setLoading] = useState(false);
  const [backendOk, setBackendOk] = useState<boolean | null>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const scrollRef = useRef<HTMLElement>(null);

  const abortRef = useRef<AbortController | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const [recording, setRecording] = useState(false);
  const [micLevel, setMicLevel] = useState(0);
  const [mics, setMics] = useState<MediaDeviceInfo[]>([]);
  const [micId, setMicId] = useState("");
  const [micHelp, setMicHelp] = useState(false);
  const [busyNote, setBusyNote] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [draft, setDraft] = useState<TranscriptDraft | null>(null);
  const [sessionRecord, setSessionRecord] = useState<SessionRecord | null>(null);

  async function startRecording(deviceId = micId) {
    setNotice(null);
    setMicHelp(false);
    try {
      window.speechSynthesis?.cancel();
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          ...(deviceId ? { deviceId: { exact: deviceId } } : {}),
        },
      });
      const devices = await navigator.mediaDevices.enumerateDevices(); // labels are only available after permission
      setMics(devices.filter((d) => d.kind === "audioinput" && d.deviceId !== "default" && d.deviceId !== "communications"));

      // Measure the real input level: it shows the student the mic is working, and lets us tell a silent mic from
      // a transcription problem (Whisper turns silence into "Thank you.").
      const audioContext = new AudioContext();
      await audioContext.resume();
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 1024;
      audioContext.createMediaStreamSource(stream).connect(analyser);
      const samples = new Float32Array(analyser.fftSize);
      let peak = 0;
      const meter = setInterval(() => {
        analyser.getFloatTimeDomainData(samples);
        const rms = Math.sqrt(samples.reduce((sum, x) => sum + x * x, 0) / samples.length);
        peak = Math.max(peak, rms);
        setMicLevel(rms);
      }, 80);

      const recorder = new MediaRecorder(stream);
      const parts: Blob[] = [];
      recorder.ondataavailable = (e) => parts.push(e.data);
      recorder.onstop = async () => {
        clearInterval(meter);
        stream.getTracks().forEach((t) => t.stop());
        audioContext.close();
        setRecording(false);
        setMicLevel(0);
        if (peak < SILENT_MIC_RMS) {
          const name = stream.getAudioTracks()[0]?.label || "the selected microphone";
          setNotice(`No sound reached the browser from "${name}". Choose another microphone below and try again.`);
          setMicHelp(true);
          return;
        }
        setBusyNote("Transcribing your question…");
        try {
          const form = new FormData();
          const type = recorder.mimeType || "audio/webm";
          form.append("file", new Blob(parts, { type }), type.includes("mp4") ? "speech.mp4" : "speech.webm");
          const res = await fetch(`${API}/voice/transcribe`, { method: "POST", body: form });
          if (!res.ok) throw new Error(await errorText(res));
          const { text } = await res.json();
          if (!text) {
            setNotice("I picked up sound but no clear words. Please try again in a quieter spot, or choose another microphone.");
            setMicHelp(true);
          }
          // goes into the box, not straight to the advisor: the student can correct a misheard course name first
          setInput((prev) => [prev, text].filter(Boolean).join(" "));
          inputRef.current?.focus();
        } catch (e) {
          setNotice(e instanceof Error ? e.message : String(e));
        } finally {
          setBusyNote(null);
        }
      };
      recorderRef.current = recorder;
      recorder.start();
      setRecording(true);
      setTimeout(() => {
        if (recorder.state === "recording") recorder.stop();
      }, 60000);
    } catch (e) {
      setRecording(false);
      if (e instanceof DOMException && e.name === "OverconstrainedError") {
        setMicId(""); // the saved microphone was unplugged
        setNotice("That microphone is no longer available. Press the mic button to use the default one.");
      } else {
        setNotice("I couldn't use the microphone. Please allow access in your browser, or type your question.");
      }
    }
  }

  async function uploadTranscript(file: File) {
    setNotice(null);
    setBusyNote("Reading your transcript…");
    try {
      const form = new FormData();
      form.append("file", await prepareImage(file), "transcript.jpg");
      const res = await fetch(`${API}/transcript/read`, { method: "POST", body: form });
      if (!res.ok) throw new Error(await errorText(res));
      const data = await res.json();
      if (!data.readable || data.courses.length === 0) {
        setNotice("I couldn't find courses and grades in that image. Please try a clearer photo of your transcript.");
      } else {
        setDraft(data);
      }
    } catch (e) {
      setNotice(e instanceof Error ? e.message : String(e));
    } finally {
      setBusyNote(null);
      if (fileRef.current) fileRef.current.value = "";
    }
  }

  function confirmDraft() {
    if (!draft?.batch || !draft.current_semester) return;
    setSessionRecord({ batch: draft.batch, current_semester: draft.current_semester, courses: draft.courses });
    setStudentId("");
    setDraft(null);
    setMessages([]);
  }

  const sessionTokens = messages.reduce((sum, m) => sum + (m.role === "assistant" ? m.reply.usage.total_tokens : 0), 0);

  // Poll /health until the backend is up (it may be started after the frontend), then load the profiles.
  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;
    async function check() {
      try {
        const r = await fetch(`${API}/health`);
        if (cancelled) return;
        setBackendOk(r.ok);
        if (r.ok) {
          const list = await fetch(`${API}/students`).then((res) => (res.ok ? res.json() : []));
          if (!cancelled) setStudents(Array.isArray(list) ? list : []);
          return;
        }
      } catch {
        if (cancelled) return;
        setBackendOk(false);
      }
      timer = setTimeout(check, 5000);
    }
    check();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, []);

  // grow the textarea with its content (Shift+Enter adds lines)
  useEffect(() => {
    const el = inputRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  }, [input]);

  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  }, [messages, loading]);

  const last = messages[messages.length - 1];
  const awaitingClarification = last?.role === "assistant" && last.reply.needs_clarification;

  // keep the input focused when the advisor asks a follow-up question
  useEffect(() => {
    if (awaitingClarification) inputRef.current?.focus();
  }, [awaitingClarification]);

  async function send(text: string) {
    const message = text.trim();
    if (!message || loading) return;
    const history = messages.flatMap((m) =>
      m.role === "user"
        ? [{ role: "user", content: m.content }]
        : m.role === "assistant"
          ? [
              {
                role: "assistant",
                content: [m.reply.answer, m.reply.clarifying_question].filter(Boolean).join(" "),
                needs_clarification: m.reply.needs_clarification,
              },
            ]
          : [],
    );
    setMessages((prev) => [...prev, { role: "user", content: message }]);
    setInput("");
    setLoading(true);
    const controller = new AbortController();
    abortRef.current = controller;
    try {
      const res = await fetch(`${API}/chat`, {
        method: "POST",
        signal: controller.signal,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message,
          strategy,
          student_id: sessionRecord ? null : studentId || null,
          session_record: sessionRecord,
          conversation_history: history,
        }),
      });
      if (!res.ok) throw new Error(await errorText(res));
      const reply: AdvisorReply = await res.json();
      setMessages((prev) => [...prev, { role: "assistant", reply }]);
      setBackendOk(true);
    } catch (e) {
      if (controller.signal.aborted) return; // "New chat" was pressed — drop the stale reply
      const network = e instanceof TypeError;
      if (network) setBackendOk(false);
      const content = network
        ? `Can't reach the advisor backend at ${API}. Is the server running?`
        : e instanceof Error
          ? e.message
          : String(e);
      setMessages((prev) => [...prev, { role: "error", content }]);
    } finally {
      if (abortRef.current === controller) {
        abortRef.current = null;
        setLoading(false);
        inputRef.current?.focus();
      }
    }
  }

  const newChat = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setLoading(false);
    setMessages([]);
    setInput("");
    setNotice(null);
    inputRef.current?.focus();
  }, []);

  const selected = students.find((s) => s.student_id === studentId);

  const selectClass =
    "w-full min-w-0 max-w-[18rem] truncate rounded-lg border border-slate-200 bg-white/90 px-2.5 py-1.5 text-sm text-slate-800 shadow-sm outline-none transition hover:border-brand-500/50 focus:border-brand-500 focus:ring-2 focus:ring-brand-100";

  return (
    <div className="relative isolate flex h-dvh flex-col overflow-hidden text-slate-900">
      {/* decorative background — fixed and static, so it never scrolls or repaints */}
      <div aria-hidden className="pointer-events-none fixed inset-0 -z-10 overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-[#f5f7ff] via-[#f8f9fc] to-[#f3f0ff]" />
        <div className="absolute -left-32 -top-32 h-96 w-96 rounded-full bg-brand-500/15 blur-3xl" />
        <div className="absolute -right-24 top-1/3 h-96 w-96 rounded-full bg-violet-400/15 blur-3xl" />
        <div className="absolute bottom-0 left-1/3 h-80 w-80 rounded-full bg-sky-300/15 blur-3xl" />
      </div>

      <header className="shrink-0 border-b border-white/60 bg-white/85 px-4 py-3 sm:px-6 shadow-sm">
        <div className="mx-auto flex max-w-5xl flex-wrap items-center gap-4">
          <div className="mr-auto flex items-center gap-4">
            <Image src="/vu-logo.png" alt="Vidyashilp University" width={110} height={40} priority />
            <div className="border-l border-slate-200 pl-4">
              <h1 className="bg-gradient-to-r from-brand-900 via-brand-600 to-violet-600 bg-clip-text text-lg font-bold tracking-tight text-transparent">
                AI Academic Advisor
              </h1>
              <div className="flex items-center gap-2 text-xs text-slate-500">
                <span
                  className={`inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 font-medium ring-1 ${
                    backendOk
                      ? "bg-emerald-50 text-emerald-700 ring-emerald-200"
                      : backendOk === false
                        ? "bg-rose-50 text-rose-700 ring-rose-200"
                        : "bg-slate-50 text-slate-500 ring-slate-200"
                  }`}
                >
                  <span className="relative flex h-2 w-2">
                    {backendOk && (
                      <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-75" />
                    )}
                    <span
                      className={`relative inline-flex h-2 w-2 rounded-full ${
                        backendOk ? "bg-emerald-500" : backendOk === false ? "bg-rose-500" : "bg-slate-400"
                      }`}
                    />
                  </span>
                  {backendOk === null ? "connecting…" : backendOk ? "online" : "offline"}
                </span>
                <span className="hidden sm:inline">Grounded in Handbook · SOP · Semester spread · Minors</span>
              </div>
            </div>
          </div>

          <label className="flex min-w-0 items-center gap-2 whitespace-nowrap text-xs font-medium text-slate-500">
            👤 Profile
            <select
              className={selectClass}
              value={studentId}
              onChange={(e) => {
                setStudentId(e.target.value);
                setSessionRecord(null);
              }}
            >
              <option value="">No profile (anonymous)</option>
              {students.map((s) => (
                <option key={s.student_id} value={s.student_id}>
                  {s.student_id}
                  {s.batch ? ` · ${s.batch} batch · sem ${s.current_semester} · CGPA ${s.cgpa}` : " · incomplete record"}
                </option>
              ))}
            </select>
          </label>
          <label className="flex min-w-0 items-center gap-2 whitespace-nowrap text-xs font-medium text-slate-500">
            🧠 Strategy
            <select
              className={selectClass}
              value={strategy}
              onChange={(e) => setStrategy(e.target.value as Strategy)}
            >
              {STRATEGIES.map((s) => (
                <option key={s.value} value={s.value}>
                  {s.label}
                </option>
              ))}
            </select>
          </label>
          <button
            className="inline-flex items-center gap-1.5 rounded-lg bg-brand-900 px-3.5 py-1.5 text-sm font-medium text-white shadow-md shadow-brand-900/20 transition hover:-translate-y-0.5 hover:bg-brand-700"
            onClick={newChat}
          >
            <span className="text-base leading-none">＋</span> New chat
          </button>
        </div>

        {sessionRecord && (
          <div className="mx-auto mt-3 flex max-w-5xl flex-wrap items-center gap-2 rounded-xl border border-emerald-200 bg-emerald-50 px-3 py-2 text-sm text-emerald-900">
            🔒 Using your confirmed record: {sessionRecord.batch} batch, semester {sessionRecord.current_semester},{" "}
            {sessionRecord.courses.length} courses. Kept in this browser tab only, never stored on the server.
            <button type="button" className="ml-auto font-medium underline" onClick={() => setSessionRecord(null)}>
              Remove
            </button>
          </div>
        )}

        {selected && !sessionRecord && (
          <div className="mx-auto mt-3 flex max-w-5xl animate-fade-up flex-wrap items-center gap-2 rounded-xl border border-brand-100 bg-gradient-to-r from-brand-50 to-violet-50 px-3 py-2">
            <div className="mr-2 flex items-center gap-2">
              <div className="flex h-8 w-8 items-center justify-center rounded-full bg-brand-900 text-xs font-bold text-white">
                {selected.student_id.slice(-2)}
              </div>
              <div>
                <div className="text-sm font-semibold text-brand-900">
                  {selected.student_id}{" "}
                  <span className="rounded bg-white/80 px-1.5 py-0.5 text-[10px] font-medium text-slate-500">
                    synthetic
                  </span>
                </div>
                <div className="max-w-md text-xs text-slate-600">{selected.scenario}</div>
              </div>
            </div>
            <div className="ml-auto flex flex-wrap gap-2">
              <Stat label="Batch" value={selected.batch} />
              <Stat label="Semester" value={selected.current_semester} />
              <Stat label="CGPA" value={selected.cgpa} />
              <Stat label="Credits" value={selected.completed_credits} />
            </div>
          </div>
        )}
      </header>

      {backendOk === false && (
        <div className="border-b border-rose-200 bg-rose-50/90 px-4 py-2 text-center text-sm text-rose-800">
          Advisor backend is offline at <span className="font-mono">{API}</span> — start it with{" "}
          <code className="rounded bg-white px-1.5 py-0.5 font-mono text-xs">
            cd backend &amp;&amp; uvicorn main:app --port 8000
          </code>
          . Retrying automatically…
        </div>
      )}

      <main ref={scrollRef} className="chat-scroll min-h-0 flex-1 overflow-y-auto overscroll-contain px-4 py-8 sm:px-6">
        <div className="mx-auto flex max-w-4xl flex-col gap-5">
          {messages.length === 0 && (
            <div className="mt-6 flex animate-fade-up flex-col items-center text-center">
              <div className="relative mb-5">
                <div className="absolute inset-0 rounded-3xl bg-gradient-to-br from-brand-500 to-violet-500 opacity-40 blur-xl" />
                <div className="relative flex h-20 w-20 items-center justify-center rounded-3xl bg-gradient-to-br from-brand-500 via-brand-700 to-brand-900 text-white shadow-xl">
                  <svg viewBox="0 0 24 24" className="h-10 w-10" fill="none" stroke="currentColor" strokeWidth={1.6}>
                    <path d="M22 10 12 5 2 10l10 5 10-5Z" strokeLinejoin="round" />
                    <path d="M6 12v5c3 2 9 2 12 0v-5" strokeLinejoin="round" />
                    <path d="M22 10v6" strokeLinecap="round" />
                  </svg>
                </div>
              </div>
              <h2 className="text-3xl font-bold tracking-tight text-slate-900 sm:text-4xl">
                Hi there! I&apos;m your{" "}
                <span className="bg-gradient-to-r from-brand-600 to-violet-600 bg-clip-text text-transparent">
                  academic advisor
                </span>
              </h2>
              <p className="mt-3 max-w-xl text-slate-600">
                Ask me about courses, prerequisites, credits, progression, attendance or registration. Every answer
                is backed by official university documents.
              </p>

              <div className="mt-8 grid w-full max-w-3xl gap-3 sm:grid-cols-2">
                {EXAMPLES.map((ex, i) => (
                  <button
                    key={ex.q}
                    onClick={() => send(ex.q)}
                    style={{ animationDelay: `${120 + i * 70}ms` }}
                    className="group flex animate-fade-up items-start gap-3 rounded-2xl border border-white/80 bg-white/80 p-4 text-left shadow-sm ring-1 ring-slate-200/70 transition hover:-translate-y-1 hover:bg-white hover:shadow-lg hover:shadow-brand-500/10 hover:ring-brand-500/40"
                  >
                    <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-brand-50 text-xl transition group-hover:scale-110">
                      {ex.icon}
                    </span>
                    <span>
                      <span className="block text-[11px] font-semibold uppercase tracking-wider text-brand-600">
                        {ex.tag}
                      </span>
                      <span className="mt-0.5 block text-sm text-slate-700">{ex.q}</span>
                    </span>
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((m, i) =>
            m.role === "user" ? (
              <div
                key={i}
                className="max-w-[85%] animate-fade-up self-end whitespace-pre-wrap break-words rounded-2xl sm:max-w-[75%] rounded-tr-md bg-gradient-to-br from-brand-500 to-brand-900 px-4 py-2.5 text-[15px] text-white shadow-lg shadow-brand-500/25"
              >
                {m.content}
              </div>
            ) : m.role === "assistant" ? (
              <div key={i} className="animate-fade-up self-start">
                <AssistantMessage
                  reply={m.reply}
                  question={messages.slice(0, i).reduce((q, x) => (x.role === "user" ? x.content : q), "")}
                />
              </div>
            ) : (
              <div
                key={i}
                className="animate-fade-up self-start rounded-xl border border-rose-200 bg-rose-50 px-4 py-2.5 text-sm text-rose-800"
              >
                ❌ Error: {m.content}
              </div>
            ),
          )}
          {loading && <TypingIndicator />}
        </div>
      </main>

      <footer className="shrink-0 px-4 pb-5 pt-2 sm:px-6">
        {draft && (
          <div className="mx-auto mb-2 max-h-[45dvh] max-w-4xl overflow-y-auto rounded-2xl border border-brand-100 bg-white p-4 shadow-xl">
            <h2 className="font-semibold text-brand-900">Check what I read from your transcript</h2>
            <p className="mt-1 text-sm text-slate-600">
              AI can misread images, so please correct anything wrong. Nothing is used until you confirm, and the image
              was not saved. Credits and CGPA are worked out from these grades and the course list.
            </p>
            {(draft.rejected.length > 0 || draft.unreadable_rows > 0) && (
              <p className="mt-2 rounded-lg bg-amber-50 px-3 py-2 text-sm text-amber-900">
                ⚠️ {draft.unreadable_rows > 0 && `${draft.unreadable_rows} row(s) were unreadable. `}
                {draft.rejected.length > 0 && `Not in the university's course list, so left out: ${draft.rejected.join(", ")}.`}
              </p>
            )}
            <div className="mt-3 flex flex-wrap gap-3 text-sm">
              <label className="flex items-center gap-2 font-medium text-slate-700">
                Batch
                <select
                  className="rounded-lg border border-slate-200 px-2 py-1"
                  value={draft.batch ?? ""}
                  onChange={(e) => setDraft({ ...draft, batch: e.target.value || null })}
                >
                  <option value="">Choose…</option>
                  {BATCHES.map((b) => (
                    <option key={b}>{b}</option>
                  ))}
                </select>
              </label>
              <label className="flex items-center gap-2 font-medium text-slate-700">
                Current semester
                <select
                  className="rounded-lg border border-slate-200 px-2 py-1"
                  value={draft.current_semester ?? ""}
                  onChange={(e) => setDraft({ ...draft, current_semester: Number(e.target.value) || null })}
                >
                  <option value="">Choose…</option>
                  {[1, 2, 3, 4, 5, 6, 7, 8].map((n) => (
                    <option key={n}>{n}</option>
                  ))}
                </select>
              </label>
            </div>
            <ul className="mt-3 grid gap-1.5 sm:grid-cols-3">
              {draft.courses.map((c, i) => (
                <li key={c.code + i} className="flex items-center gap-2 rounded-lg bg-slate-50 px-2 py-1 text-sm">
                  <span className="font-mono text-xs">{c.code}</span>
                  <select
                    aria-label={`Grade for ${c.code}`}
                    className="ml-auto rounded border border-slate-200 bg-white px-1"
                    value={c.grade}
                    onChange={(e) =>
                      setDraft({
                        ...draft,
                        courses: draft.courses.map((x, j) => (j === i ? { ...x, grade: e.target.value } : x)),
                      })
                    }
                  >
                    {GRADES.map((g) => (
                      <option key={g}>{g}</option>
                    ))}
                  </select>
                  <button
                    type="button"
                    aria-label={`Remove ${c.code}`}
                    className="text-slate-400 hover:text-rose-600"
                    onClick={() => setDraft({ ...draft, courses: draft.courses.filter((_, j) => j !== i) })}
                  >
                    ✕
                  </button>
                </li>
              ))}
            </ul>
            <div className="mt-3 flex items-center gap-2">
              <button
                type="button"
                disabled={!draft.batch || !draft.current_semester}
                onClick={confirmDraft}
                className="rounded-lg bg-brand-900 px-3.5 py-1.5 text-sm font-medium text-white disabled:opacity-40"
              >
                Use this record
              </button>
              <button type="button" onClick={() => setDraft(null)} className="rounded-lg px-3 py-1.5 text-sm text-slate-600 hover:bg-slate-100">
                Discard
              </button>
              {(!draft.batch || !draft.current_semester) && (
                <span className="text-sm text-amber-800">Please choose your batch and semester.</span>
              )}
            </div>
          </div>
        )}
        {recording && (
          <div className="mx-auto mb-2 flex max-w-4xl items-center gap-3 rounded-lg bg-rose-50 px-3 py-1.5 text-sm text-rose-900">
            <span className="whitespace-nowrap">🎙️ Listening… speak, then press ⏹</span>
            <div
              role="meter"
              aria-label="Microphone level"
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={Math.round(Math.min(1, micLevel * 12) * 100)}
              className="h-2 flex-1 overflow-hidden rounded-full bg-rose-100"
            >
              <div className="h-full bg-rose-500 transition-[width]" style={{ width: `${Math.min(1, micLevel * 12) * 100}%` }} />
            </div>
          </div>
        )}
        {(notice || busyNote) && (
          <p className={`mx-auto mb-2 max-w-4xl rounded-lg px-3 py-1.5 text-sm ${notice ? "bg-rose-50 text-rose-800" : "bg-brand-50 text-brand-900"}`}>
            {notice ?? busyNote}
          </p>
        )}
        {micHelp && mics.length > 0 && (
          <label className="mx-auto mb-2 flex max-w-4xl items-center gap-2 rounded-lg bg-amber-50 px-3 py-1.5 text-sm font-medium text-amber-900">
            🎤 Microphone
            <select
              className="min-w-0 flex-1 rounded-lg border border-amber-200 bg-white px-2 py-1 font-normal"
              value={micId}
              onChange={(e) => {
                setMicId(e.target.value);
                startRecording(e.target.value);
              }}
            >
              <option value="">System default</option>
              {mics.map((m) => (
                <option key={m.deviceId} value={m.deviceId}>
                  {m.label || "Microphone"}
                </option>
              ))}
            </select>
          </label>
        )}
        <form
          className={`mx-auto flex max-w-4xl items-end gap-2 rounded-2xl border bg-white p-2 shadow-xl transition focus-within:ring-4 ${
            awaitingClarification
              ? "border-brand-500/60 shadow-brand-500/20 ring-brand-100"
              : "border-slate-200 shadow-slate-300/40 focus-within:border-brand-500/50 focus-within:ring-brand-100"
          }`}
          onSubmit={(e) => {
            e.preventDefault();
            send(input);
          }}
        >
          <input
            ref={fileRef}
            type="file"
            accept="image/png,image/jpeg,image/webp"
            className="hidden"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) uploadTranscript(file);
            }}
          />
          <button
            type="button"
            title="Upload a photo of your transcript (a person's name or ID on it is not kept)"
            aria-label="Upload transcript image"
            disabled={!!busyNote}
            onClick={() => fileRef.current?.click()}
            className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl text-lg text-slate-500 transition hover:bg-brand-50 disabled:opacity-40"
          >
            📎
          </button>
          <textarea
            ref={inputRef}
            rows={1}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
                e.preventDefault();
                send(input);
              }
            }}
            placeholder={awaitingClarification ? "Answer the advisor's question…" : "Ask an academic question…"}
            className="max-h-40 flex-1 resize-none bg-transparent px-3 py-2.5 text-[15px] outline-none placeholder:text-slate-400"
          />
          <button
            type="button"
            title={recording ? "Stop recording" : "Ask by voice. You can check the text before sending."}
            aria-label={recording ? "Stop recording" : "Record a voice question"}
            aria-pressed={recording}
            disabled={!!busyNote}
            onClick={() => (recording ? recorderRef.current?.stop() : startRecording())}
            className={`flex h-11 w-11 shrink-0 items-center justify-center rounded-xl text-lg transition disabled:opacity-40 ${
              recording ? "animate-pulse bg-rose-100 text-rose-700" : "text-slate-500 hover:bg-brand-50"
            }`}
          >
            {recording ? "⏹" : "🎤"}
          </button>
          <button
            type="submit"
            disabled={loading || !input.trim()}
            aria-label="Send"
            className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-gradient-to-br from-brand-500 to-brand-900 text-white shadow-md shadow-brand-500/30 transition hover:scale-105 disabled:scale-100 disabled:opacity-35 disabled:shadow-none"
          >
            <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth={2}>
              <path d="M5 12h14M13 6l6 6-6 6" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </button>
        </form>
        <p className="mx-auto mt-2 max-w-4xl text-center text-[11px] text-slate-400">
          AI-generated guidance: always confirm important decisions with your Faculty Advisor · Enter to send · This
          conversation: {sessionTokens.toLocaleString()} tokens
        </p>
      </footer>
    </div>
  );
}
