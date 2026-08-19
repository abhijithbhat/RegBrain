"use client";

import React, { useEffect, useRef, useState } from "react";
import {
  ArrowRight,
  BookOpen,
  Database,
  FileText,
  Moon,
  RefreshCw,
  Search,
  ShieldCheck,
  Sun,
  XCircle,
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
    desc: "Locating relevant passages in the RBI Master Directions index.",
    stepNumber: "01",
  },
  reranking: {
    label: "Cross-encoder review",
    desc: "Ranking candidate clauses against the inquiry on record.",
    stepNumber: "02",
  },
  generating: {
    label: "Statutory synthesis",
    desc: "Drafting a grounded finding with explicit clause attribution.",
    stepNumber: "03",
  },
  verifying: {
    label: "Dual-gate verification",
    desc: "Checking lexical fidelity and neural entailment before entry.",
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
    label: "Know Your Customer — NBFCs",
    query: "What are the KYC requirements for NBFCs?",
  },
  {
    label: "Capital adequacy — commercial banks",
    query: "What is the capital adequacy requirement for Commercial Banks?",
  },
  {
    label: "Comparative KYC records",
    query: "Compare the KYC requirements between Commercial Banks and NBFCs",
  },
  {
    label: "Dividend distribution — NBFCs",
    query: "What are the dividend distribution rules for NBFCs?",
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

  const apiUrl = process.env.NEXT_PUBLIC_API_URL || "https://regbrain.onrender.com";
  const apiKey = process.env.NEXT_PUBLIC_API_KEY || "regbrain-dev-key";

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
      let receivedResult = false;
      try {
        const response = await fetch(`${apiUrl}/query/stream`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-API-Key": apiKey,
          },
          body: JSON.stringify({ question: trimmed, session_id: sessionId || null }),
          signal: abortController.signal,
        });

        if (response.ok && response.body) {
          const reader = response.body.getReader();
          const decoder = new TextDecoder();
          let buffer = "";

          while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const parts = buffer.split("\n\n");
            buffer = parts.pop() || "";

            for (const part of parts) {
              const record = part.trim();
              if (!record.startsWith("data:")) continue;

              try {
                const data = JSON.parse(record.replace(/^data:\s*/, ""));
                if (data.stage) {
                  setCurrentStage(data.stage as StageType);
                } else if (data.status || data.answer !== undefined) {
                  setResult(data);
                  if (data.session_id) setSessionId(data.session_id);
                  setCurrentStage(null);
                  setIsLoading(false);
                  receivedResult = true;
                } else if (data.error) {
                  setErrorMessage(data.message || data.error);
                  setCurrentStage(null);
                  setIsLoading(false);
                  receivedResult = true;
                }
              } catch (parseError) {
                console.error("Error parsing SSE JSON chunk:", parseError, record);
              }
            }
          }
        }
      } catch (streamError) {
        console.warn("Stream interrupted, falling back to standard endpoint...", streamError);
      }

      // If streaming was interrupted or did not deliver final result, fallback to standard /query
      if (!receivedResult) {
        const fallbackRes = await fetch(`${apiUrl}/query`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-API-Key": apiKey,
          },
          body: JSON.stringify({ question: trimmed, session_id: sessionId || null }),
          signal: abortController.signal,
        });

        if (!fallbackRes.ok) {
          let detail = `HTTP ${fallbackRes.status}`;
          try {
            const errorBody = await fallbackRes.json();
            detail = errorBody.detail?.message || errorBody.detail || errorBody.message || detail;
          } catch {
            // Preserve status code
          }
          throw new Error(detail);
        }

        const data = await fallbackRes.json();
        setResult(data);
        if (data.session_id) setSessionId(data.session_id);
      }
    } catch (error: unknown) {
      if (!(error instanceof Error && error.name === "AbortError")) {
        setErrorMessage(
          error instanceof Error
            ? error.message || "Failed to communicate with RegBrain audit service"
            : "Failed to communicate with RegBrain audit service"
        );
      }
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
        <aside className="ledger-spine" aria-label="Audit controls">
          <span className="spine-vertical-rule" aria-hidden="true" />

          <div className="spine-brand">
            <span className="crest-mark" aria-hidden="true">R</span>
            <div>
              <p className="structural-label">RegBrain</p>
              <p className="mt-1 font-serif text-lg leading-none text-[var(--text-primary)]">Regulatory Ledger</p>
            </div>
          </div>

          <div className="binding-rule" aria-hidden="true" />

          <section aria-label="Current audit session">
            <p className="structural-label">Audit session</p>
            <div className="mt-2 flex items-center justify-between gap-2">
              <span className="session-id text-xs text-[var(--text-primary)]">
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

          <section className="mt-8" aria-label="Quick questions">
            <p className="structural-label">Quick questions</p>
            <nav className="registry-list">
              {quickQuestions.map((item, index) => (
                <button
                  className="registry-item"
                  disabled={isLoading}
                  key={item.query}
                  onClick={() => setQuestion(item.query)}
                  type="button"
                >
                  <span className="registry-index">{String(index + 1).padStart(2, "0")}</span>
                  <span>{item.label}</span>
                </button>
              ))}
            </nav>
          </section>

          <div className="spine-controls">
            <button
              className="ledger-control"
              onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
              title={theme === "dark" ? "Switch to light parchment mode" : "Switch to dark ledger mode"}
              type="button"
            >
              {theme === "dark" ? <Sun className="h-3 w-3" /> : <Moon className="h-3 w-3" />}
              {theme === "dark" ? "Parchment" : "Ledger"}
            </button>
          </div>
        </aside>

        <div className="ledger-page-wrap">
          <main className="ledger-page ledger-opening">
            <header>
              <div className="masthead-topline">
                <span>{referenceNumber}</span>
                <span className="masthead-dateline">{currentDate || "15 August 2026"}</span>
              </div>
              <p className="masthead-department">Department of Regulatory Intelligence &amp; Compliance</p>
              <h1 className="masthead-title">The Regulatory Ledger</h1>
              <p className="masthead-summary">
                Notifications, obligations, and audit findings entered for deliberate review.
              </p>
              <p className="rbi-scope">
                <span>RBI regulatory scope</span>
                Built solely for Reserve Bank of India directions, circulars, and notifications.
              </p>
            </header>

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
                    placeholder="Ask a regulatory question…"
                    type="text"
                    value={question}
                  />
                  <button className="ledger-submit" disabled={isLoading || !question.trim()} type="submit">
                    {isLoading ? "Auditing" : "Enter"}
                    {!isLoading && <ArrowRight className="ml-1 inline h-3.5 w-3.5" />}
                  </button>
                </div>
                <div className="query-footer">
                  <span id="query-title">RBI-only scope · Master Directions, circulars &amp; notifications</span>
                  <span className="flex gap-3">
                    <span>Dual-gate verification</span>
                    <button className="query-clear" disabled={isLoading || !question} onClick={() => setQuestion("")} type="button">
                      Clear entry
                    </button>
                  </span>
                </div>
              </form>
            </section>

            {isLoading && (
              <section className="ledger-section" aria-live="polite">
                <div className="section-heading">
                  <h2 className="section-title">Audit in progress</h2>
                  <span className="session-id text-xs text-[var(--accent-brass)]">LIVE SSE</span>
                </div>
                <div className="audit-progress" aria-label="Audit progress">
                  <div className="audit-progress-steps">
                    {STAGES_ORDER.map((stageName, index) => {
                      const stage = STAGE_CONFIG[stageName];
                      const state = activeStageIndex > index ? "is-complete" : currentStage === stageName ? "is-current" : "";
                      return (
                        <div className={`audit-progress-step ${state}`} key={stageName}>
                          <span className="audit-progress-step-number">{stage.stepNumber}</span>
                          <span>{stage.label}</span>
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
                    <p className="entry-kicker">Currently running</p>
                    <h3>{activeStage.label}</h3>
                    <p>{activeStage.desc}</p>
                  </div>
                  <span className="audit-active-stage-status">RBI source review</span>
                </article>
              </section>
            )}

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
                        <p className="entry-prose">{result.answer || "No findings recorded."}</p>
                        <div className="entry-meta">
                          {result.was_decomposed && <span>Multi-hop inquiry reconciled</span>}
                          {result.was_rewritten && <span>Follow-up context resolved</span>}
                          {!result.was_decomposed && !result.was_rewritten && <span>Single inquiry review</span>}
                        </div>
                      </div>
                    </article>

                    {result.citations && result.citations.length > 0 && <CitationsList citations={result.citations} />}
                  </>
                )}
              </section>
            )}

            {!result && !isLoading && !errorMessage && (
              <section className="ledger-section" aria-labelledby="standby-title">
                <div className="section-heading">
                  <h2 className="section-title" id="standby-title">Ready for audit</h2>
                  <span className="flex items-center gap-1.5 text-xs text-[var(--accent-supported)]">
                    <span className="h-1.5 w-1.5 rounded-full bg-[var(--accent-supported)]" /> Ready for entry
                  </span>
                </div>
                <article className="ledger-entry">
                  <span className="folio-number">00</span>
                  <div>
                    <p className="entry-kicker">Standing instruction</p>
                    <h3 className="entry-title">Begin with a specific regulatory question.</h3>
                    <p className="entry-prose">
                      RegBrain retrieves applicable RBI Master Directions, verifies candidate clauses, and records only grounded findings.
                    </p>
                    <div className="entry-meta">
                      <span className="flex items-center gap-1.5"><ShieldCheck className="h-3.5 w-3.5 text-[var(--accent-supported)]" /> Lexical &amp; NLI gates</span>
                      <span className="flex items-center gap-1.5"><Database className="h-3.5 w-3.5 text-[var(--accent-brass)]" /> Indexed RBI corpus</span>
                    </div>
                  </div>
                </article>
              </section>
            )}

            <footer className="mt-16 border-t border-[var(--border-hairline)] pt-4 text-[0.65rem] tracking-[0.04em] text-[var(--text-secondary)]">
              RegBrain · RBI regulatory intelligence · Source-grounded audit record
            </footer>
          </main>
        </div>
      </div>
    </div>
  );
}
