"""
run_audit.py
Runs answer_query() across 5 test questions and prints/saves full results.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from generation.answer_query import answer_query

TEST_QUESTIONS = [
    "What are the KYC requirements for NBFCs?",
    "What is the capital adequacy requirement for Commercial Banks?",
    "What are the dividend distribution rules for NBFCs?",
    "What are the exposure limits for a single borrower?",
    "What are the rules for digital lending?",
]

def main():
    results = []
    output_file = "generation/audit_results.json"

    for i, q in enumerate(TEST_QUESTIONS, 1):
        print(f"\n=== Processing Q{i}: '{q}' ===")
        res = answer_query(q)
        results.append({
            "q_num": i,
            "query": q,
            "result": res
        })
        if i < len(TEST_QUESTIONS):
            print("Sleeping 12s to respect Groq rate limits...")
            time.sleep(12)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone! Results saved to {output_file}")

if __name__ == "__main__":
    main()
