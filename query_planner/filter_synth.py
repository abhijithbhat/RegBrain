import re
from generation.verify import _split_into_sentences, _get_nli_model

def filter_ungrounded_sentences(synthesized_text: str, citations: list[dict]) -> str:
    """Ensure every sentence in synthesized_text is backed by at least one citation."""
    if not synthesized_text or not citations:
        return synthesized_text
    
    sentences = _split_into_sentences(synthesized_text)
    if not sentences:
        return synthesized_text
        
    citation_texts = [c.get("evidence_sentence", c.get("text", "")) for c in citations if isinstance(c, dict)]
    citation_blob = " ".join(citation_texts).lower()
    
    # Quick lexical + NLI verification
    nli_model = _get_nli_model()
    valid_sentences = []
    
    for sent in sentences:
        # Check if the sentence has strong lexical overlap with citations
        sent_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", sent.lower()))
        if not sent_words:
            continue
        overlap = len(sent_words.intersection(set(re.findall(r"\b[a-zA-Z]{4,}\b", citation_blob)))) / len(sent_words)
        
        # If strong overlap (>= 0.50), keep it
        if overlap >= 0.50:
            valid_sentences.append(sent)
        else:
            # Check NLI entailment against citation texts
            pairs = [[ct, sent] for ct in citation_texts]
            if pairs:
                scores = nli_model.predict(pairs)
                # scores is [entailment, neutral, contradiction]
                best_entailment = max(s[0] for s in scores)
                best_contradiction = min(s[2] for s in scores)
                if best_entailment > 0.0:
                    valid_sentences.append(sent)
                else:
                    print(f"DROPPING UNGROUNDED SYNTHESIS SENTENCE:\n  \"{sent}\"\n  (overlap={overlap:.2f}, best_entailment={best_entailment:.2f})")
    
    return " ".join(valid_sentences)
