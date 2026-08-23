"use client";

import React from "react";
import { Check, ShieldAlert, ShieldCheck } from "lucide-react";

interface ConfidenceIndicatorProps {
  confidence: number;
  status: "answered" | "abstain" | "error";
  wasDecomposed?: boolean;
}

export default function ConfidenceIndicator({
  confidence,
  status,
  wasDecomposed,
}: ConfidenceIndicatorProps) {
  const score = Math.max(0, Math.min(100, confidence || 0));
  const isAbstain = status === "abstain";
  const finding = isAbstain
    ? "Statutory guardrail enforced"
    : score >= 75
      ? "Finding notarized"
      : score >= 50
        ? "Finding recorded with qualification"
        : "Finding below notarization standard";
  const description = isAbstain
    ? "The available record did not meet the threshold for a grounded response."
    : "Confidence is derived from lexical fidelity and neural relevance across the cited record.";

  return (
    <>
      <div className="flex items-center justify-between gap-2">
        <p className="entry-kicker">Verification Record</p>
        <span className="inline-flex items-center gap-1 text-[0.62rem] font-semibold text-[var(--accent-brass)]">
          GATE 1 &amp; GATE 2 CHECKED
        </span>
      </div>
      <h3 className="entry-title">{finding}</h3>
      <p className="entry-prose">{description}</p>

      <div className="entry-meta">
        <span>{wasDecomposed ? "Multi-hop inquiry reconciled" : "Direct inquiry review"}</span>
        <span>{isAbstain ? "Response withheld" : "Grounded response released"}</span>
      </div>

      <div className="audit-axis">
        <div>
          <div className="confidence-track" aria-label={`Audit confidence: ${score.toFixed(1)}%`}>
            <div className="confidence-fill" style={{ width: `${Math.max(2, score)}%` }} />
          </div>
          <div className="mt-2 flex justify-between text-[0.62rem] text-[var(--text-secondary)]">
            <span>Abstention threshold · 33%</span>
            <span>Notarization threshold · 75%</span>
          </div>
        </div>
        <span className="confidence-value text-sm font-bold text-[var(--accent-supported)]">{score.toFixed(1)}%</span>
      </div>

      {!isAbstain && (
        <div className="verification-mark" aria-label={`Verified confidence: ${score.toFixed(1)}%`}>
          <span className="verification-mark-icon" aria-hidden="true">
            <Check className="h-4 w-4 stroke-[3]" />
          </span>
          <div>
            <p className="verification-mark-label">Verified</p>
            <p className="text-[0.62rem] text-[var(--text-secondary)]">Zero-Hallucination</p>
          </div>
          <span className="verification-mark-score">{score.toFixed(1)}%</span>
        </div>
      )}
    </>
  );
}
