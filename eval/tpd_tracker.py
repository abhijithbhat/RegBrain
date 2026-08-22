"""
eval/tpd_tracker.py – Tracks Tokens Per Day (TPD) usage and runtime pacing during evaluations.
"""
import time


class TPDTracker:
    def __init__(self):
        self.start_time = time.time()
        self.total_tokens_used = 0

    def record_usage(self, tokens: int):
        self.total_tokens_used += tokens

    def log_status(self, current: int, total: int):
        elapsed = time.time() - self.start_time
        print(f"  [TPD Tracker] Progress: {current}/{total} completed ({elapsed:.1f}s elapsed)", flush=True)

    def summary(self, stage: str = "eval"):
        elapsed = time.time() - self.start_time
        print(f"  [TPD Tracker] Stage '{stage}' finished in {elapsed:.1f}s.", flush=True)


tpd_tracker = TPDTracker()
