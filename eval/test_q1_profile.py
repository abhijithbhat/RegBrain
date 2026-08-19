import time
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from query_planner.handle_query import handle_query

print("=== RUNNING QUESTION 1 PROFILER ===", flush=True)
q1 = "What are the KYC requirements for NBFCs?"
session_state = {}

t0 = time.time()
print(f"[{time.time()-t0:.2f}s] Starting handle_query...", flush=True)
out = handle_query(q1, session_state)
print(f"[{time.time()-t0:.2f}s] handle_query finished!", flush=True)
print(json.dumps(out, indent=2, default=str), flush=True)
