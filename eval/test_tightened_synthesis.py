import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from query_planner.plan_and_answer import plan_and_answer

test_qs = [
    "What are the governance guidelines for commercial bank boards and what are the credit facility limits for NBFCs?",
    "What are the rules for digital lending?"
]

for idx, q in enumerate(test_qs, 2):
    print(f"\n========================================================", flush=True)
    print(f"CASE {idx}: \"{q}\"", flush=True)
    print(f"========================================================\n", flush=True)
    t0 = time.time()
    res = plan_and_answer(q)
    print(f"Elapsed: {time.time()-t0:.2f}s", flush=True)
    print(f"Status: {res.get('status')}", flush=True)
    print(f"Confidence: {res.get('confidence')}%", flush=True)
    print(f"\nFINAL SYNTHESIZED ANSWER:\n{res.get('answer')}\n", flush=True)
    
    citations = res.get("citations", [])
    print(f"TOTAL VERIFIED CLAIMS ({len(citations)}):", flush=True)
    for j, c in enumerate(citations, 1):
        print(f"  {j}. [{c.get('doc_id')} | {c.get('source_clause_id')}] \"{c.get('text')}\"", flush=True)
    
    time.sleep(2)
