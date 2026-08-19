"""
eval/tpd_tracker.py — Lightweight daily token budget tracker for Groq API.

Tracks cumulative tokens consumed across all API calls in a session.
Prints budget status every N questions and at the end.

Usage:
    from eval.tpd_tracker import tpd_tracker
    tpd_tracker.add(prompt_tokens=500, completion_tokens=200)
    tpd_tracker.log_status(question_idx=5)
"""

import os

DAILY_BUDGET = int(os.getenv("GROQ_TPD_BUDGET", "200000"))


class TPDTracker:
    """Tracks cumulative token usage against a daily budget."""

    def __init__(self, budget: int = DAILY_BUDGET):
        self.budget = budget
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.api_calls = 0
        self.rate_limit_hits = 0
        self.first_tpd_error_idx: int | None = None

    def add(self, prompt_tokens: int = 0, completion_tokens: int = 0,
            question_idx: int | None = None) -> None:
        """Record tokens consumed by an API call."""
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens += (prompt_tokens + completion_tokens)
        self.api_calls += 1

    def record_rate_limit(self, question_idx: int | None = None) -> None:
        """Record a 429 TPD rate-limit error."""
        self.rate_limit_hits += 1
        if self.first_tpd_error_idx is None and question_idx is not None:
            self.first_tpd_error_idx = question_idx

    def log_status(self, question_idx: int, total_questions: int) -> None:
        """Print budget status (call every 5 questions)."""
        pct = (self.total_tokens / self.budget * 100) if self.budget > 0 else 0
        remaining = self.budget - self.total_tokens
        print(f"    📊 TPD Budget: {self.total_tokens:,}/{self.budget:,} tokens used "
              f"({pct:.1f}%) | {remaining:,} remaining | "
              f"{self.api_calls} API calls | "
              f"Q{question_idx}/{total_questions}")

    def summary(self, stage: str = "unknown") -> None:
        """Print final summary."""
        pct = (self.total_tokens / self.budget * 100) if self.budget > 0 else 0
        print(f"\n  📊 TPD BUDGET SUMMARY ({stage}):")
        print(f"     Prompt tokens:     {self.prompt_tokens:>10,}")
        print(f"     Completion tokens: {self.completion_tokens:>10,}")
        print(f"     Total tokens:      {self.total_tokens:>10,} / {self.budget:,} ({pct:.1f}%)")
        print(f"     API calls:         {self.api_calls:>10,}")
        print(f"     429 rate-limit hits: {self.rate_limit_hits:>8,}")
        if self.first_tpd_error_idx is not None:
            print(f"     First TPD error at: Q{self.first_tpd_error_idx}")
        print()


# Global singleton for the current session
tpd_tracker = TPDTracker()
