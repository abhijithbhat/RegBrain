"use client";

import React from "react";

interface VerificationSealProps {
  size?: "sm" | "md" | "lg" | "xl";
  className?: string;
  withText?: boolean;
}

export default function VerificationSeal({
  size = "sm",
  className = "",
  withText = false,
}: VerificationSealProps) {
  const dimensions = {
    sm: "h-7 w-7",
    md: "h-10 w-10",
    lg: "h-16 w-16",
    xl: "h-32 w-32",
  };

  return (
    <div
      className={`relative inline-flex shrink-0 items-center justify-center ${dimensions[size]} ${className}`}
      title="Notarized verification seal"
    >
      <svg
        aria-hidden="true"
        className="h-full w-full text-[var(--accent-seal)]"
        fill="none"
        viewBox="0 0 96 96"
        xmlns="http://www.w3.org/2000/svg"
      >
        <circle cx="48" cy="48" fill="currentColor" fillOpacity="0.12" r="45" />
        <circle cx="48" cy="48" r="43" stroke="currentColor" strokeDasharray="3 2" strokeWidth="2" />
        <circle cx="48" cy="48" r="37" stroke="currentColor" strokeWidth="1.2" />
        <circle cx="48" cy="48" r="28" stroke="currentColor" strokeWidth="1.5" />
        <path d="M48 20l2.2 5.8L56 28l-5.8 2.2L48 36l-2.2-5.8L40 28l5.8-2.2L48 20Z" fill="currentColor" />
        <path d="M48 60l2.2 5.8L56 68l-5.8 2.2L48 76l-2.2-5.8L40 68l5.8-2.2L48 60Z" fill="currentColor" />
        <path d="M31 49l10 9 24-26" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="5" />
        <text fill="currentColor" fontFamily="IBM Plex Mono, monospace" fontSize="6" letterSpacing="1.4" textAnchor="middle" x="48" y="85">
          VERIFIED
        </text>
      </svg>
      {withText && (
        <span className="ml-2 whitespace-nowrap text-[0.62rem] font-semibold tracking-[0.1em] text-[var(--accent-seal)]">
          VERIFIED
        </span>
      )}
    </div>
  );
}
