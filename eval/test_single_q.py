import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from query_planner.plan_and_answer import plan_and_answer
from retrieval.retrieve import retrieve

print("Step 1: Test retrieve...", flush=True)
t0 = time.time()
chunks = retrieve("What reporting must banks submit to RBI?")
print(f"Retrieved {len(chunks)} chunks in {time.time()-t0:.2f}s", flush=True)

print("Step 2: Test plan_and_answer...", flush=True)
t0 = time.time()
res = plan_and_answer("What reporting must banks submit to RBI?")
print(f"Done in {time.time()-t0:.2f}s", flush=True)
print("Result status:", res.get("status"), flush=True)
print("Result answer:", res.get("answer"), flush=True)
print("Result citations count:", len(res.get("citations", [])), flush=True)
