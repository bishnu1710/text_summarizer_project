import os
from src.pdf_utils import extract_text_from_pdf, chunk_text

def test_chunk_text():
    text = "A" * 7000
    chunks = chunk_text(text, max_length=3000)
    assert len(chunks) == 3
    assert all(isinstance(chunk, str) for chunk in chunks)

# PDF extraction test (skipped unless sample.pdf exists)
def test_extract_text_from_pdf():
    if os.path.exists("data/sample.pdf"):
        text = extract_text_from_pdf("data/sample.pdf")
        assert isinstance(text, str)
