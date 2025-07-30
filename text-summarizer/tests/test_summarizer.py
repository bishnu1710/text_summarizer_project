from src.summarizer import Summarizer

def test_summarizer_initialization():
    summarizer = Summarizer("facebook/bart-large-cnn")
    assert summarizer is not None

def test_summarizer_output():
    summarizer = Summarizer("facebook/bart-large-cnn")
    summary = summarizer.summarize("AI is transforming the world of technology.")
    assert isinstance(summary, str)
    assert len(summary) > 0
