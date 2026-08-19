import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from query_planner.plan_and_answer import plan_and_answer, classify_query, decompose_query
from retrieval.retrieve import retrieve

questions = [
    "What reporting must banks submit to RBI?",
    "What are the governance guidelines for commercial bank boards and what are the credit facility limits for NBFCs?",
    "What are the rules for digital lending?"
]

results = []

for idx, q in enumerate(questions, 1):
    print(f"\n{'=' * 80}", flush=True)
    print(f"QUESTION {idx}: \"{q}\"", flush=True)
    print(f"{'=' * 80}\n", flush=True)
    
    # 1. Classification & Retrieval
    classification = classify_query(q)
    print(f"Planner Classification: needs_decomposition={classification.get('needs_decomposition')} | reasoning: {classification.get('reasoning')}", flush=True)
    
    if classification.get("needs_decomposition"):
        sub_qs = decompose_query(q)
        print(f"Decomposed Sub-queries ({len(sub_qs)}):", flush=True)
        for sq in sub_qs:
            print(f"  - {sq}", flush=True)
            sq_chunks = retrieve(sq)
            print(f"    Top chunks for \"{sq}\":", flush=True)
            for c_i, c in enumerate(sq_chunks[:3], 1):
                doc = c.get("doc_id", "")
                cid = c.get("clause_id", "")
                snip = c.get("clause_text", "").strip().replace("\n", " ")[:100]
                print(f"      {c_i}. [{doc} | {cid}] {snip}...", flush=True)
    else:
        chunks = retrieve(q)
        print(f"\nTop Retrieved Chunks for Main Query ({len(chunks)}):", flush=True)
        for c_i, c in enumerate(chunks, 1):
            doc = c.get("doc_id", "")
            cid = c.get("clause_id", "")
            snip = c.get("clause_text", "").strip().replace("\n", " ")[:120]
            print(f"  {c_i}. [{doc} | {cid}] {snip}...", flush=True)
    
    print("\n--- RUNNING plan_and_answer() ---", flush=True)
    t0 = time.time()
    res = plan_and_answer(q)
    elapsed = time.time() - t0
    
    print(f"\nExecution time: {elapsed:.2f}s", flush=True)
    print(f"Status: {res.get('status')}", flush=True)
    print(f"Was Decomposed: {res.get('was_decomposed')}", flush=True)
    print(f"Confidence: {res.get('confidence'):.1f}%", flush=True)
    print(f"\nFULL ANSWER:\n{res.get('answer')}\n", flush=True)
    
    citations = res.get("citations", [])
    print(f"TOTAL CLAIMS / CITATIONS: {len(citations)}", flush=True)
    for j, c in enumerate(citations, 1):
        print(f"\n  [Claim {j}]", flush=True)
        print(f"    • Claim Text: \"{c.get('text')}\"", flush=True)
        print(f"    • Cited Clause ID: {c.get('cited_clause_id')} (Doc: {c.get('doc_id')}, Source Clause: {c.get('source_clause_id')})", flush=True)
        print(f"    • Evidence Sentence: \"{c.get('evidence_sentence')}\"", flush=True)
        print(f"    • Supported: {c.get('supported')} (lexical_pass={c.get('lexical_pass')}, nli_pass={c.get('nli_pass')})", flush=True)
        print(f"    • Verifier Metrics: kw_score={c.get('kw_score')}, seq_score={c.get('seq_score')}, nums_ok={c.get('nums_ok')}", flush=True)
        if c.get("best_entailment") is not None:
            print(f"    • NLI: ent={c.get('best_entailment')}, con={c.get('best_contradiction')}", flush=True)
    
    results.append({
        "question_id": idx,
        "question": q,
        "classification": classification,
        "result": res
    })
    time.sleep(2)

with open("eval/validation_3_questions_output.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, default=str)

print("\nSaved full validation outputs to eval/validation_3_questions_output.json", flush=True)
