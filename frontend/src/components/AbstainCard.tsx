"use client";

import React from "react";

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
        <p className="entry-kicker text-[var(--accent-abstain)]">Grounding threshold not met</p>
        <h3 className="entry-title">No notarized finding has been entered.</h3>
        <blockquote className="source-excerpt border-[var(--accent-abstain)] text-[var(--text-primary)]">“{explanation}”</blockquote>
        <p className="entry-prose text-[var(--text-secondary)]">
          The inquiry remains on record without a generated regulatory assertion, preserving the statutory guardrail.
        </p>
        <div className="entry-meta">
          <span>Inquiry on record: “{question}”</span>
          <span className="confidence-value text-[var(--accent-abstain)]">0.0%</span>
        </div>
      </div>
    </article>
  );
}
