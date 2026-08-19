import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from query_planner.plan_and_answer import plan_and_answer

q = "What are the rules for digital lending?"
print(f"Starting plan_and_answer for: {q}", flush=True)
t0 = time.time()
res = plan_and_answer(q)
print(f"Finished in {time.time()-t0:.2f}s", flush=True)
print("Result JSON:", flush=True)
print(json.dumps(res, indent=2, default=str), flush=True)
