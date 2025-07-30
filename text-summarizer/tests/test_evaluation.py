from src.evaluation import evaluate_summary

def test_evaluate_summary():
    reference = "AI is changing the world."
    generated = "AI is transforming the world."
    scores = evaluate_summary(reference, generated)
    assert "ROUGE-1" in scores
    assert 0 <= scores["ROUGE-1"] <= 1
