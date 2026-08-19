import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "RegBrain — The Regulatory Ledger",
  description: "Official RBI Compliance Audit Ledger backed by Two-Stage Neural Claim Verification & Notarized Grounding.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen antialiased selection:bg-[#A98953]/30 selection:text-[#EDE6D6]">
        {children}
      </body>
    </html>
  );
}
