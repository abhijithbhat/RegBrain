import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from query_planner.plan_and_answer import plan_and_answer

print("==================================================================", flush=True)
print("CASE 2: Governance guidelines for bank boards & NBFC credit limits", flush=True)
print("==================================================================", flush=True)
q2 = "What are the governance guidelines for commercial bank boards and what are the credit facility limits for NBFCs?"
t0 = time.time()
res2 = plan_and_answer(q2)
print(f"Elapsed: {time.time()-t0:.2f}s", flush=True)
print(f"Status: {res2.get('status')}", flush=True)
print(f"Confidence: {res2.get('confidence')}%", flush=True)
print(f"\nSYNTHESIZED ANSWER (CASE 2):\n{res2.get('answer')}\n", flush=True)
print(f"VERIFIED CITATIONS COUNT: {len(res2.get('citations', []))}", flush=True)
for i, c in enumerate(res2.get('citations', []), 1):
    print(f"  {i}. [{c.get('doc_id')} | {c.get('source_clause_id')}] (supported={c.get('supported')}) \"{c.get('text')}\"", flush=True)

print("\nSleeping 5s to avoid rate limit spikes...\n", flush=True)
time.sleep(5)

print("==================================================================", flush=True)
print("CASE 3: Rules for digital lending", flush=True)
print("==================================================================", flush=True)
q3 = "What are the rules for digital lending?"
t0 = time.time()
res3 = plan_and_answer(q3)
print(f"Elapsed: {time.time()-t0:.2f}s", flush=True)
print(f"Status: {res3.get('status')}", flush=True)
print(f"Confidence: {res3.get('confidence')}%", flush=True)
print(f"\nSYNTHESIZED ANSWER (CASE 3):\n{res3.get('answer')}\n", flush=True)
print(f"VERIFIED CITATIONS COUNT: {len(res3.get('citations', []))}", flush=True)
for i, c in enumerate(res3.get('citations', []), 1):
    print(f"  {i}. [{c.get('doc_id')} | {c.get('source_clause_id')}] (supported={c.get('supported')}) \"{c.get('text')}\"", flush=True)

with open("eval/tightened_validation_output.json", "w", encoding="utf-8") as f:
    json.dump({"case2": res2, "case3": res3}, f, indent=2, default=str)
print("\nSaved output to eval/tightened_validation_output.json", flush=True)
