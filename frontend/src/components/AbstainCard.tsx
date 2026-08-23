"use client";

import React from "react";
import { ShieldAlert } from "lucide-react";

interface AbstainCardProps {
  question: string;
  reason?: string;
  answer?: string;
}

export default function AbstainCard({ question, reason, answer }: AbstainCardProps) {
  const explanation =
    reason ||
    answer ||
    "The official regulatory corpus does not contain sufficiently verified clauses to answer this inquiry with zero-hallucination fidelity.";

  return (
    <article className="ledger-entry exception-entry">
      <span className="folio-number text-[var(--accent-abstain)]">01</span>
      <div>
        <div className="flex items-center gap-2">
          <p className="entry-kicker text-[var(--accent-abstain)]">Grounding threshold not met</p>
          <span className="inline-flex items-center gap-1 text-[0.62rem] font-semibold text-[var(--accent-abstain)] bg-[var(--accent-abstain)]/10 px-1.5 py-0.5 rounded">
            <ShieldAlert className="h-3 w-3" /> GUARDRAIL ENFORCED
          </span>
        </div>
        <h3 className="entry-title">No notarized finding has been entered.</h3>
        <blockquote className="source-excerpt border-[var(--accent-abstain)] text-[var(--text-primary)]">“{explanation}”</blockquote>
        <p className="entry-prose text-[var(--text-secondary)]">
          The inquiry remains on record without a generated regulatory assertion, preventing hallucination and preserving the strict statutory guardrail.
        </p>
        <div className="entry-meta">
          <span>Inquiry on record: “{question}”</span>
          <span className="confidence-value text-[var(--accent-abstain)] font-bold">0.0% (Abstained)</span>
        </div>
      </div>
    </article>
  );
}
