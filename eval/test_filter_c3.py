import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
from query_planner.filter_synth import filter_ungrounded_sentences

with open("eval/tightened_validation_output.json") as f:
    data = json.load(f)

c3 = data["case3"]
orig_answer = c3["answer"]
citations = c3["citations"]

print("=== ORIGINAL CASE 3 SYNTHESIZED ANSWER ===")
print(orig_answer)
print("\n=== FILTERING UNGROUNDED SENTENCES ===")
filtered = filter_ungrounded_sentences(orig_answer, citations)
print("\n=== FILTERED ANSWER ===")
print(filtered)
