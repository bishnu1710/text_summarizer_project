from rouge_score import rouge_scorer

def evaluate_summary(reference, generated):
    """
    Compute ROUGE scores between reference and generated summaries.

    Args:
        reference (str): Reference (ground-truth) summary.
        generated (str): Generated summary by model.
    Returns:
        dict: ROUGE-1, ROUGE-2, and ROUGE-L scores.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
    scores = scorer.score(reference, generated)

    return {
        "ROUGE-1": round(scores['rouge1'].fmeasure, 3),
        "ROUGE-2": round(scores['rouge2'].fmeasure, 3),
        "ROUGE-L": round(scores['rougeL'].fmeasure, 3)
    }
