# RegBrain vs Naive Baseline — Evaluation Comparison

Evaluated on **38 regulatory questions** covering NBFCs, commercial banks, Small Finance Banks, and Payments Banks.

## Metric Comparison

| Metric | RegBrain | Naive Baseline | Delta | Winner |
|--------|----------|----------------|-------|--------|
| Faithfulness (all) | 1.0000 | 0.9786 | +0.0214 | RegBrain ✅ |
| Faithfulness (answered) | 1.0000 | 0.9786 | +0.0214 | RegBrain ✅ |
| Answer Relevancy (all) | 0.6288 | 0.8270 | -0.1982 | Naive ⚠️ |
| Answer Relevancy (ans.) | 0.8850 | 0.8270 | +0.0580 | RegBrain ✅ |
| Context Precision | 0.5200 | 0.4867 | +0.0333 | RegBrain ✅ |
| Context Recall | 0.7500 | 0.5000 | +0.2500 | RegBrain ✅ |
| Hit Rate (overall %) | 0.0 | 40.0 | -40.0 | Naive ⚠️ |
| Hit Rate (answered %) | 0.0 | 40.0 | -40.0 | Naive ⚠️ |
| Hallucination Rate (%) | 0.0 | 2.9 | -2.9 | RegBrain ✅ |
| Confidence (all %) | 54.76 | 82.86 | -28.10 | Naive ⚠️ |
| Confidence (ans. %) | 77.07 | 82.86 | -5.79 | Naive ⚠️ |
| Citations / Answer (all) | 2.87 | 1.71 | +1.15 | RegBrain ✅ |
| Citations / Answer (ans.) | 4.04 | 1.71 | +2.32 | RegBrain ✅ |
| Answer Rate (%) | 71.1 | 100.0 | -28.9 | Naive ⚠️ |

## Hallucination Rates (Side-by-Side)

> Note: Verifier and RAGAS are two independent groundedness checks operating at different levels of rigor.

| Check Level | RegBrain | Naive Baseline |
|-------------|----------|----------------|
| Verifier (lexical + entailment margin) | 13/27 (48.1%) | 0/35 (0.0%) |
| RAGAS faithfulness < 1.0 | 0/27 (0.0%) | 1/35 (2.9%) |

## Key Takeaways

- **Faithfulness (answered only)**: RegBrain achieves **1.00** vs Naive's **0.98** — a **1.0×** improvement in answer groundedness.
- **Hallucination Rate (RAGAS)**: RegBrain has **0.0%** hallucination rate vs Naive's **2.9%**.
- **Hallucination Rate (Verifier)**: RegBrain has **48.1%** verifier hallucination rate.
- **Hit Rate (answered only)**: RegBrain correctly cites the expected clause **0.0%** of the time (overall: 0.0%) vs Naive's **40.0%**.
- **Answer Relevancy (answered only)**: RegBrain scores **0.89** vs Naive's **0.83** — the overall near-tie (0.54 vs 0.54) is an artifact of abstain answers getting 0.0 relevancy, dragging RegBrain's average down.
- **Answer Rate Trade-off**: RegBrain answers 27/38 (71.1%) — it abstains when evidence is insufficient. Naive answers 35/35 (100.0%) but with significantly lower faithfulness.

## Scoring Methodology Note

- **Abstained questions are included in all overall ("all") averages.** Abstains receive faithfulness=1.0 (not hallucinating is faithful) and answer_relevancy=0.0 ('insufficient evidence' is not topically relevant). This slightly inflates RegBrain's overall faithfulness and depresses its overall relevancy.
- **"Answered only" metrics** provide the apples-to-apples comparison on the subset of questions each pipeline actually answered.
- **Hit Rate (overall)** treats abstains as automatic misses (conservative). **Hit Rate (answered)** uses only answered questions as denominator (more accurate).

## Status Disagreements

11 questions where RegBrain abstains but Naive answers:

| ID | RegBrain | Naive | Question |
|----|----------|-------|----------|
| 9 | abstain | answered | What is the SLR requirement for banks? |
| 16 | abstain | answered | What are the KYC requirements and capital adequacy norms for |
| 19 | abstain | answered | How did the Priority Sector Lending (PSL) guidelines change  |
| 22 | abstain | answered | What are the rules regarding liquidity risk management and L |
| 25 | abstain | answered | What are the prompt corrective action (PCA) framework trigge |
| 27 | abstain | answered | What is the minimum priority sector lending requirement for  |
| 29 | abstain | answered | What is the maximum loan size restriction for Small Finance  |
| 30 | abstain | answered | What is the maximum balance limit per individual customer in |
| 31 | abstain | answered | Are Payments Banks allowed to undertake lending activities o |
| 32 | abstain | answered | Where must Payments Banks invest their customer demand depos |
| 33 | abstain | answered | What are the customer protection and grievance redressal mec |
