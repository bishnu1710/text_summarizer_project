import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    """
    Extracts all text from a PDF file.
    
    Args:
        pdf_path (str): Path to the PDF file.
    Returns:
        str: All extracted text from the PDF.
    """
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text, max_length=3000):
    """
    Splits long text into smaller chunks so transformer models can process them.
    
    Args:
        text (str): Input text to split.
        max_length (int): Maximum number of characters per chunk.
    Returns:
        list: List of text chunks.
    """
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]
