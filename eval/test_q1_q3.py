import time
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from query_planner.handle_query import handle_query

qs = [
    "What are the KYC requirements for NBFCs?",
    "What is the capital adequacy requirement for Commercial Banks?",
    "What are the penalties for non-compliance?"
]

for idx, q in enumerate(qs, 1):
    print(f"\n--- Running Q{idx}: {q} ---", flush=True)
    t0 = time.time()
    out = handle_query(q, {})
    print(f"Done in {time.time()-t0:.2f}s | Status: {out.get('status')} | Conf: {out.get('confidence')}%", flush=True)
