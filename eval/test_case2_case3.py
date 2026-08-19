import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from query_planner.plan_and_answer import plan_and_answer

print("=== RUNNING CASE 2 ===", flush=True)
q2 = "What are the governance guidelines for commercial bank boards and what are the credit facility limits for NBFCs?"
t0 = time.time()
res2 = plan_and_answer(q2)
print(f"Done in {time.time()-t0:.2f}s", flush=True)
print("Result 2:", flush=True)
print(json.dumps(res2, indent=2, default=str), flush=True)

print("\nSleeping 5s to respect TPM rate limit...\n", flush=True)
time.sleep(5)

print("=== RUNNING CASE 3 ===", flush=True)
q3 = "What are the rules for digital lending?"
t0 = time.time()
res3 = plan_and_answer(q3)
print(f"Done in {time.time()-t0:.2f}s", flush=True)
print("Result 3:", flush=True)
print(json.dumps(res3, indent=2, default=str), flush=True)
