"use client";

import React, { useEffect, useRef, useState } from "react";
import {
  ArrowRight,
  Database,
  Moon,
  RefreshCw,
  Search,
  ShieldCheck,
  Sparkles,
  Sun,
  X,
} from "lucide-react";
import AbstainCard from "@/components/AbstainCard";
import CitationsList, { CitationItem } from "@/components/CitationsList";
import ConfidenceIndicator from "@/components/ConfidenceIndicator";

interface QueryResult {
  status: "answered" | "abstain" | "error";
  answer?: string;
  citations?: CitationItem[];
  confidence?: number;
  reason?: string;
  category?: string;
  was_decomposed?: boolean;
  was_rewritten?: boolean;
  original_query?: string;
  rewritten_query?: string;
  session_id?: string;
}

type StageType = "retrieving" | "reranking" | "generating" | "verifying" | null;

const STAGE_CONFIG: Record<
  Exclude<StageType, null>,
  { label: string; desc: string; stepNumber: string }
> = {
  retrieving: {
    label: "Dense & sparse retrieval",
    desc: "Locating relevant passages across the 105 RBI Master Directions index.",
    stepNumber: "01",
  },
  reranking: {
    label: "Cross-encoder review",
    desc: "Ranking candidate clauses against the statutory inquiry on record.",
    stepNumber: "02",
  },
  generating: {
    label: "Statutory synthesis",
    desc: "Drafting a grounded finding with explicit clause attribution.",
    stepNumber: "03",
  },
  verifying: {
    label: "Dual-gate verification",
    desc: "Checking lexical fidelity and neural relevance before notary entry.",
    stepNumber: "04",
  },
};

const STAGES_ORDER: Exclude<StageType, null>[] = [
  "retrieving",
  "reranking",
  "generating",
  "verifying",
];

const quickQuestions = [
  {
    chip: "Commercial Banks CRAR",
    label: "Capital Adequacy — Commercial Banks",
    query: "What is the capital adequacy requirement for Commercial Banks?",
  },
  {
    chip: "UCB Prudential Norms",
    label: "Prudential Norms & CRAR — Urban Co-op Banks (UCB)",
    query: "What are the capital adequacy norms for Urban Co-operative Banks?",
  },
  {
    chip: "HFC Guidelines",
    label: "Housing Finance Companies (HFC) Guidelines",
    query: "What are the regulatory guidelines for Housing Finance Companies (HFCs)?",
  },
  {
    chip: "NBFC Dividend Rules",
    label: "Dividend Distribution — NBFCs",
    query: "What are the dividend distribution rules for NBFCs?",
  },
  {
    chip: "3-Way Capital Norms Comparison",
    label: "3-Way Comparative: Capital Norms (Banks vs NBFCs vs UCBs)",
    query: "Compare the capital adequacy requirements between Commercial Banks, NBFCs, and Urban Co-operative Banks.",
  },
];

function formatCurrentDate() {
  return new Intl.DateTimeFormat("en-GB", {
    day: "numeric",
    month: "long",
    year: "numeric",
    timeZone: "Asia/Kolkata",
  }).format(new Date());
}

export default function Home() {
  const [question, setQuestion] = useState("");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [currentStage, setCurrentStage] = useState<StageType>(null);
  const [result, setResult] = useState<QueryResult | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [theme, setTheme] = useState<"dark" | "light">("dark");
  const [currentDate, setCurrentDate] = useState("");
  const abortControllerRef = useRef<AbortController | null>(null);

  const apiUrl = "/api/backend";

  useEffect(() => {
    document.documentElement.classList.toggle("dark", theme === "dark");
  }, [theme]);

  useEffect(() => {
    const updateDate = () => setCurrentDate(formatCurrentDate());
    updateDate();
    const interval = window.setInterval(updateDate, 60_000);
    return () => window.clearInterval(interval);
  }, []);

  const handleResetSession = () => {
    setSessionId(null);
    setResult(null);
    setErrorMessage(null);
    setCurrentStage(null);
  };

  const handleQuickQuestionSelect = (query: string) => {
    setQuestion(query);
  };

  const handleSubmit = async (event?: React.FormEvent) => {
    event?.preventDefault();
    const trimmed = question.trim();
    if (!trimmed || isLoading) return;

    setIsLoading(true);
    setCurrentStage("retrieving");
    setErrorMessage(null);
    setResult(null);

    abortControllerRef.current?.abort();
    const abortController = new AbortController();
    abortControllerRef.current = abortController;

    try {
      // 1. Start the asynchronous query job
      let startData: any = null;
      const MAX_START_ATTEMPTS = 3;

      for (let attempt = 1; attempt <= MAX_START_ATTEMPTS; attempt++) {
        try {
          const res = await fetch(`${apiUrl}/query/start`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query: trimmed, session_id: sessionId || undefined }),
            signal: abortController.signal,
          });

          if (res.ok) {
            startData = await res.json();
            break;
          }

          if (res.status === 502 || res.status === 503 || res.status === 504) {
            if (attempt < MAX_START_ATTEMPTS) {
              await new Promise((r) => setTimeout(r, 2500));
              continue;
            }
          }

          const errJson = await res.json().catch(() => ({}));
          throw new Error(errJson.detail || `Server responded with status ${res.status}`);
        } catch (err: any) {
          if (err.name === "AbortError") throw err;
          if (attempt === MAX_START_ATTEMPTS) throw err;
          await new Promise((r) => setTimeout(r, 2000));
        }
      }

      if (!startData || !startData.job_id) {
        throw new Error("Failed to initialize query job with RegBrain service.");
      }

      const jobId = startData.job_id;

      // 2. Poll for job completion
      const POLL_INTERVAL_MS = 600;
      const MAX_POLL_TIME_MS = 90_000;
      const startTime = Date.now();

      while (Date.now() - startTime < MAX_POLL_TIME_MS) {
        if (abortController.signal.aborted) return;

        const pollRes = await fetch(`${apiUrl}/query/result/${jobId}`, {
          signal: abortController.signal,
        });

        if (!pollRes.ok) {
          throw new Error(`Failed to check query status: ${pollRes.status}`);
        }

        const pollData = await pollRes.json();

        if (pollData.stage && STAGES_ORDER.includes(pollData.stage)) {
          setCurrentStage(pollData.stage);
        }

        if (pollData.status === "completed") {
          const finalRes = pollData.result || {};
          setResult({
            status: finalRes.status || "answered",
            answer: finalRes.answer,
            citations: finalRes.citations || [],
            confidence: finalRes.confidence,
            reason: finalRes.reason,
            category: finalRes.category,
            was_decomposed: finalRes.was_decomposed,
            was_rewritten: finalRes.was_rewritten,
            original_query: finalRes.original_query,
            rewritten_query: finalRes.rewritten_query,
            session_id: finalRes.session_id || sessionId || undefined,
          });
          if (finalRes.session_id) {
            setSessionId(finalRes.session_id);
          }
          return;
        }

        if (pollData.status === "failed") {
          throw new Error(pollData.error || "Query execution failed.");
        }

        await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS));
      }

      throw new Error("Query processing timed out. Please try again.");
    } catch (error) {
      if (error instanceof Error && error.name === "AbortError") {
        return;
      }
      setErrorMessage(
        error instanceof Error
          ? error.message || "Failed to communicate with RegBrain audit service"
          : "Failed to communicate with RegBrain audit service"
      );
    } finally {
      setIsLoading(false);
      setCurrentStage(null);
    }
  };

  const activeStageIndex = currentStage ? STAGES_ORDER.indexOf(currentStage) : 0;
  const activeStage = currentStage ? STAGE_CONFIG[currentStage] : STAGE_CONFIG.retrieving;
  const referenceNumber = sessionId
    ? `No. RB/REG/${sessionId.slice(0, 8).toUpperCase()}`
    : "No. RB/REG/2026/0815";

  return (
    <div className="ledger-shell">
      <div className="ledger-layout">
        {/* Left Spine Sidebar (Desktop & Tablet Navigation) */}
        <aside className="ledger-spine" aria-label="Audit controls">
          <span className="spine-vertical-rule" aria-hidden="true" />

          <div className="spine-brand">
            <span className="crest-mark" aria-hidden="true">R</span>
            <div>
              <p className="structural-label">RegBrain</p>
              <p className="mt-0.5 font-serif text-lg leading-none text-[var(--text-primary)]">The Regulatory Ledger</p>
            </div>
          </div>

          <div className="binding-rule" aria-hidden="true" />

          <section aria-label="Current audit session" className="hidden lg:block">
            <p className="structural-label">Audit session</p>
            <div className="mt-2 flex items-center justify-between gap-2">
              <span className="session-id text-xs text-[var(--accent-brass)]">
                {sessionId ? sessionId.slice(0, 12) : "UNOPENED"}
              </span>
              {sessionId && (
                <button className="ledger-control" onClick={handleResetSession} title="Reset audit session">
                  <RefreshCw className="h-3 w-3" />
                  Reset
                </button>
              )}
            </div>
          </section>

          <section className="mt-6 hidden lg:block" aria-label="Quick questions">
            <p className="structural-label">Preset Inquiries</p>
            <nav className="registry-list">
              {quickQuestions.map((item, index) => (
                <button
                  className="registry-item cursor-pointer"
                  disabled={isLoading}
                  key={item.query}
                  onClick={() => handleQuickQuestionSelect(item.query)}
                  type="button"
                >
                  <span className="registry-index">{String(index + 1).padStart(2, "0")}</span>
                  <span>{item.label}</span>
                </button>
              ))}
            </nav>
          </section>

          <div className="spine-controls">
            {sessionId && (
              <button className="ledger-control lg:hidden" onClick={handleResetSession} title="Reset audit session">
                <RefreshCw className="h-3 w-3" />
                Reset
              </button>
            )}
            <button
              className="ledger-control cursor-pointer"
              onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
              title={theme === "dark" ? "Switch to light parchment mode" : "Switch to dark ledger mode"}
              type="button"
            >
              {theme === "dark" ? <Sun className="h-3 w-3 text-amber-300" /> : <Moon className="h-3 w-3 text-emerald-800" />}
              <span>{theme === "dark" ? "Parchment" : "Vault"}</span>
            </button>
          </div>
        </aside>

        {/* Main Gazette & Audit Findings Page */}
        <div className="ledger-page-wrap">
          <main className="ledger-page ledger-opening">
            <header>
              <div className="masthead-topline">
                <span>{referenceNumber}</span>
                <span className="masthead-dateline">{currentDate || "23 August 2026"}</span>
              </div>
              <p className="masthead-department">Department of Regulatory Intelligence &amp; Compliance</p>
              <h1 className="masthead-title">The Regulatory Ledger</h1>
              <p className="masthead-summary">
                Notifications, obligations, and statutory findings entered for deliberate regulatory review.
              </p>
              <div className="rbi-scope">
                <span>RBI Regulatory Scope</span>
                <p>Commercial Banks · NBFCs · Housing Finance (HFCs) · Urban Co-operative Banks (UCBs) · 105 Master Directions (15,351 Clauses)</p>
              </div>
            </header>

            {/* Gilded Search Desk Console */}
            <section className="ledger-entry-line" aria-labelledby="query-title">
              <form onSubmit={handleSubmit}>
                <div className="query-line">
                  <Search className="h-4 w-4 shrink-0 text-[var(--accent-brass)]" aria-hidden="true" />
                  <input
                    aria-labelledby="query-title"
                    className="query-field"
                    disabled={isLoading}
                    onChange={(event) => setQuestion(event.target.value)}
                    onKeyDown={(event) => {
                      if (event.key === "Enter" && !event.shiftKey) {
                        event.preventDefault();
                        handleSubmit();
                      }
                    }}
                    placeholder="Compare capital adequacy or ask a regulatory question…"
                    type="text"
                    value={question}
                  />
                  {question && !isLoading && (
                    <button
                      className="text-[var(--text-secondary)] hover:text-[var(--text-primary)] cursor-pointer mr-1 p-1"
                      onClick={() => setQuestion("")}
                      title="Clear text"
                      type="button"
                    >
                      <X className="h-3.5 w-3.5" />
                    </button>
                  )}
                  <button className="ledger-submit" disabled={isLoading || !question.trim()} type="submit">
                    {isLoading ? (
                      <span className="flex items-center gap-1.5 text-[var(--accent-supported)]">
                        <span className="h-2 w-2 animate-ping rounded-full bg-[var(--accent-supported)]" /> Auditing
                      </span>
                    ) : (
                      <>
                        <span>Enter</span>
                        <ArrowRight className="ml-1 inline h-3.5 w-3.5 text-[var(--accent-brass)]" />
                      </>
                    )}
                  </button>
                </div>

                <div className="query-footer">
                  <span id="query-title">RBI-only scope · Master Directions, circulars &amp; notifications</span>
                  <span className="flex items-center gap-3">
                    <span className="flex items-center gap-1 text-[var(--accent-supported)]">
                      <ShieldCheck className="h-3 w-3" /> Dual-gate verification
                    </span>
                  </span>
                </div>
              </form>

              {/* Horizontal Touch-Friendly Inquiry Chips */}
              <div className="inquiry-chips-wrap" aria-label="Preset regulatory inquiries">
                <span className="text-[0.63rem] font-semibold tracking-wider text-[var(--accent-brass)] uppercase shrink-0">
                  Quick Inquiries:
                </span>
                {quickQuestions.map((item, idx) => (
                  <button
                    className="inquiry-chip"
                    disabled={isLoading}
                    key={item.chip}
                    onClick={() => handleQuickQuestionSelect(item.query)}
                    type="button"
                  >
                    <span className="inquiry-chip-num">0{idx + 1}</span>
                    <span>{item.chip}</span>
                  </button>
                ))}
              </div>
            </section>

            {/* Live Audit Stepper */}
            {isLoading && (
              <section className="ledger-section" aria-live="polite">
                <div className="section-heading">
                  <h2 className="section-title">Audit in progress</h2>
                  <span className="session-id text-xs text-[var(--accent-supported)] flex items-center gap-1.5">
                    <span className="h-2 w-2 rounded-full bg-[var(--accent-supported)] animate-pulse" /> LIVE AUDIT RUNNER
                  </span>
                </div>
                <div className="audit-progress" aria-label="Audit progress">
                  <div className="audit-progress-steps">
                    {STAGES_ORDER.map((stageName, index) => {
                      const stage = STAGE_CONFIG[stageName];
                      const state = activeStageIndex > index ? "is-complete" : currentStage === stageName ? "is-current" : "";
                      return (
                        <div className={`audit-progress-step ${state}`} key={stageName}>
                          <span className="audit-progress-step-number">{stage.stepNumber}</span>
                          <span className="truncate">{stage.label}</span>
                        </div>
                      );
                    })}
                  </div>
                  <div className="audit-progress-rail" aria-label={`Audit stage ${activeStageIndex + 1} of ${STAGES_ORDER.length}`} role="list">
                    {STAGES_ORDER.map((stageName, index) => {
                      const state = activeStageIndex > index ? "is-complete" : currentStage === stageName ? "is-current" : "";
                      return <span aria-hidden="true" className={`audit-progress-segment ${state}`} key={stageName} role="listitem" />;
                    })}
                  </div>
                </div>
                <article className="audit-active-stage">
                  <span className="audit-active-stage-number">{activeStage.stepNumber}/04</span>
                  <div>
                    <p className="entry-kicker">Active verification gate</p>
                    <h3>{activeStage.label}</h3>
                    <p>{activeStage.desc}</p>
                  </div>
                  <span className="audit-active-stage-status">RBI Source Verification</span>
                </article>
              </section>
            )}

            {/* Error / Exception Display */}
            {errorMessage && (
              <section className="ledger-section" aria-live="assertive">
                <article className="ledger-entry exception-entry">
                  <span className="folio-number text-[var(--accent-abstain)]">!</span>
                  <div>
                    <p className="entry-kicker text-[var(--accent-abstain)]">Audit exception</p>
                    <h2 className="entry-title">The requested entry could not be recorded.</h2>
                    <p className="entry-prose text-[var(--text-secondary)]">{errorMessage}</p>
                  </div>
                </article>
              </section>
            )}

            {/* Notarized Audit Findings Output */}
            {result && (
              <section className="ledger-section" aria-labelledby="findings-title">
                <div className="section-heading">
                  <h2 className="section-title" id="findings-title">Findings entered</h2>
                  <span className="structural-label">Audit folio</span>
                </div>

                {result.status === "abstain" ? (
                  <AbstainCard question={question} reason={result.reason} answer={result.answer} />
                ) : (
                  <>
                    <article className="ledger-entry audit-entry">
                      <span className="folio-number">01</span>
                      <div className="audit-copy">
                        <ConfidenceIndicator
                          confidence={result.confidence || 0}
                          status={result.status}
                          wasDecomposed={result.was_decomposed}
                        />
                      </div>
                    </article>

                    <article className="ledger-entry">
                      <span className="folio-number">02</span>
                      <div>
                        <p className="entry-kicker">Synthesized statutory finding</p>
                        <h3 className="entry-title">Finding on the inquiry recorded above</h3>
                        <p className="entry-prose font-normal">{result.answer || "No findings recorded."}</p>
                        <div className="entry-meta flex flex-wrap gap-2 items-center">
                          {result.category && (
                            <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-semibold uppercase tracking-wider bg-[var(--accent-brass)]/15 text-[var(--accent-brass)] border border-[var(--accent-brass)]/30">
                              {result.category.replace(/_/g, " ")}
                            </span>
                          )}
                          {result.was_decomposed && <span className="text-[var(--accent-brass)] font-medium">Multi-hop inquiry reconciled</span>}
                          {result.was_rewritten && <span className="text-[var(--accent-brass)] font-medium">Follow-up context resolved</span>}
                          {!result.was_decomposed && !result.was_rewritten && <span>Single inquiry review</span>}
                        </div>
                      </div>
                    </article>

                    {result.citations && result.citations.length > 0 && <CitationsList citations={result.citations} />}
                  </>
                )}
              </section>
            )}

            {/* Standby / Initial State */}
            {!result && !isLoading && !errorMessage && (
              <section className="ledger-section" aria-labelledby="standby-title">
                <div className="section-heading">
                  <h2 className="section-title" id="standby-title">Ready for audit</h2>
                  <span className="flex items-center gap-1.5 text-xs text-[var(--accent-supported)] font-medium">
                    <span className="h-1.5 w-1.5 rounded-full bg-[var(--accent-supported)] animate-ping" /> System Ready
                  </span>
                </div>
                <article className="ledger-entry">
                  <span className="folio-number">00</span>
                  <div>
                    <p className="entry-kicker">Standing instruction</p>
                    <h3 className="entry-title">Begin with a specific regulatory question.</h3>
                    <p className="entry-prose">
                      RegBrain retrieves applicable RBI Master Directions across Commercial Banks, NBFCs, and UCBs, verifies candidate clauses through neural and lexical gates, and records only grounded findings.
                    </p>
                    <div className="entry-meta">
                      <span className="flex items-center gap-1.5"><ShieldCheck className="h-3.5 w-3.5 text-[var(--accent-supported)]" /> Dual-gate verification</span>
                      <span className="flex items-center gap-1.5"><Database className="h-3.5 w-3.5 text-[var(--accent-brass)]" /> 105 Master Directions</span>
                      <span className="flex items-center gap-1.5"><Sparkles className="h-3.5 w-3.5 text-[var(--accent-brass-light)]" /> Zero Hallucination Standard</span>
                    </div>
                  </div>
                </article>
              </section>
            )}

            <footer className="mt-16 border-t border-[var(--border-hairline)] pt-4 text-[0.65rem] tracking-[0.05em] text-[var(--text-tertiary)] flex justify-between items-center flex-wrap gap-2">
              <span>RegBrain · RBI Regulatory Intelligence &amp; Compliance</span>
              <span>Source-Grounded Audit Record · Dual-Gate Verified</span>
            </footer>
          </main>
        </div>
      </div>
    </div>
  );
}
