import argparse
from src.pdf_utils import extract_text_from_pdf, chunk_text
from src.summarizer import Summarizer

def main():
    parser = argparse.ArgumentParser(description="Summarize PDF using Transformer models")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("--model", default="facebook/bart-large-cnn",
                        help="Transformer model to use (default: BART)")

    args = parser.parse_args()

    #  Initialize summarizer
    summarizer = Summarizer(args.model)

    #  Extract text
    text = extract_text_from_pdf(args.pdf_path)

    #  Chunk and summarize
    chunks = chunk_text(text)
    summaries = [summarizer.summarize(chunk) for chunk in chunks]
    final_summary = " ".join(summaries)

    print("\n📜 FINAL SUMMARY:\n")
    print(final_summary)

if __name__ == "__main__":
    main()
