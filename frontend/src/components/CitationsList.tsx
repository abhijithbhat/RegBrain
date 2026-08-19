"use client";

import React, { useState } from "react";
import { ChevronDown, ChevronUp } from "lucide-react";

export interface CitationItem {
  text: string;
  cited_clause_id: string;
  source_clause_id?: string;
  doc_id?: string;
  category?: string;
  supported?: boolean;
  lexical_pass?: boolean;
  nli_pass?: boolean;
  seq_score?: number;
  kw_score?: number;
  nums_ok?: boolean;
  best_sentence?: string;
  evidence_sentence?: string;
  best_entailment?: number;
  best_contradiction?: number;
  reason?: string;
}

interface CitationsListProps {
  citations: CitationItem[];
}

export default function CitationsList({ citations }: CitationsListProps) {
  const [expandedIndex, setExpandedIndex] = useState<number | null>(null);

  if (citations.length === 0) return null;

  return (
    <section className="mt-4" aria-labelledby="citations-title">
      <div className="section-heading">
        <h3 className="section-title" id="citations-title">Verified statutory clauses</h3>
        <span className="session-id text-xs text-[var(--text-secondary)]">{citations.length} RECORDS</span>
      </div>

      {citations.map((citation, index) => {
        const isExpanded = expandedIndex === index;
        const excerpt = citation.evidence_sentence || citation.best_sentence || citation.text;
        const isSupported = citation.supported !== false;
        const title = citation.category || citation.doc_id || "RBI Master Directions";

        return (
          <article className="ledger-entry" key={`${citation.cited_clause_id}-${index}`}>
            <span className="folio-number">{String(index + 3).padStart(2, "0")}</span>
            <div>
              <p className="entry-kicker">{isSupported ? "Supported source record" : "Unverified source record"}</p>
              <h4 className="entry-title">{title}</h4>
              <p className="entry-prose">{citation.text}</p>
              <blockquote className="source-excerpt">“{excerpt}”</blockquote>

              <div className="entry-meta">
                <span className="session-id">§ {citation.source_clause_id || citation.cited_clause_id}</span>
                <span>{isSupported ? "Entailment supported" : "Entailment not established"}</span>
                <button
                  aria-expanded={isExpanded}
                  className="entry-action"
                  onClick={() => setExpandedIndex(isExpanded ? null : index)}
                  type="button"
                >
                  {isExpanded ? "Close audit particulars" : "View audit particulars"}
                  {isExpanded ? <ChevronUp className="ml-1 inline h-3 w-3" /> : <ChevronDown className="ml-1 inline h-3 w-3" />}
                </button>
              </div>

              {isExpanded && (
                <div className="telemetry-grid">
                  <div className="telemetry-cell">
                    <p className="telemetry-label">Lexical overlap</p>
                    <p className="telemetry-value">{citation.kw_score !== undefined ? `${(citation.kw_score * 100).toFixed(1)}%` : "Not recorded"}</p>
                    <p className="mt-1 text-[0.65rem] text-[var(--text-secondary)]">Sequence: {citation.seq_score !== undefined ? `${(citation.seq_score * 100).toFixed(1)}%` : "—"}</p>
                  </div>
                  <div className="telemetry-cell">
                    <p className="telemetry-label">Numeric fidelity</p>
                    <p className={`telemetry-value ${citation.nums_ok === false ? "text-[var(--accent-abstain)]" : ""}`}>
                      {citation.nums_ok === false ? "Mismatch observed" : "Verbatim record"}
                    </p>
                  </div>
                  <div className="telemetry-cell">
                    <p className="telemetry-label">Neural entailment</p>
                    <p className="telemetry-value">{citation.best_entailment !== undefined ? citation.best_entailment.toFixed(3) : "Not recorded"}</p>
                    <p className="mt-1 text-[0.65rem] text-[var(--text-secondary)]">Contradiction: {citation.best_contradiction !== undefined ? citation.best_contradiction.toFixed(3) : "—"}</p>
                  </div>
                </div>
              )}
            </div>
          </article>
        );
      })}
    </section>
  );
}
