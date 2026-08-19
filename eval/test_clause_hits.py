import json

with open("eval/raw_outputs_regbrain.json") as f:
    records = json.load(f)

hits = 0
total = 0
for r in records:
    exp = r.get("expected_clause_id", "").strip().lower()
    citations = r.get("citations", [])
    hit = 0
    if exp and citations:
        for c in citations:
            # Check source_clause_id, clause_id, doc_id
            c_source = str(c.get("source_clause_id", "")).strip().lower()
            c_label = str(c.get("clause_label", "")).strip().lower()
            c_id = str(c.get("clause_id", "")).strip().lower()
            c_doc = str(c.get("doc_id", "")).strip().lower()
            
            candidates = [c_source, c_label, c_id, c_doc]
            if any(exp == cand or exp in cand or cand in exp for cand in candidates if cand):
                hit = 1
                break
    if exp:
        total += 1
        if hit:
            hits += 1
        print(f"Q{r['question_id']}: Expected='{exp}' | Hit={hit} | Sources={[c.get('source_clause_id') for c in citations]}")

print(f"\nTotal expected clauses: {total}")
print(f"Total hits: {hits}")
print(f"Correct Clause Hit Rate: {hits / total * 100:.1f}%")
