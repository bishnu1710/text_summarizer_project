import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pdf_utils import extract_text_from_pdf, chunk_text
from src.summarizer import Summarizer
from io import BytesIO
from reportlab.pdfgen import canvas   # for PDF download
from reportlab.lib.pagesizes import letter
import gradio as gr

def summarize_pdf(pdf_file, model_choice):
    """
    Summarizes an uploaded PDF using the selected model.
    Returns summary text, downloadable TXT bytes, and downloadable PDF bytes.
    """

    # ✅ Save uploaded PDF temporarily
    temp_path = "temp.pdf"
    with open(temp_path, "wb") as f:
        f.write(pdf_file.read())

    # ✅ Extract and process text
    text = extract_text_from_pdf(temp_path)
    chunks = chunk_text(text)

    summarizer = Summarizer(model_choice)

    summaries = []
    for chunk in chunks:
        summary = summarizer.summarize(chunk)
        summaries.append(summary)

    final_summary = " ".join(summaries)

    # ✅ Create TXT download
    txt_bytes = BytesIO()
    txt_bytes.write(final_summary.encode('utf-8'))
    txt_bytes.seek(0)

    # ✅ Create PDF download
    pdf_bytes = BytesIO()
    c = canvas.Canvas(pdf_bytes, pagesize=letter)
    width, height = letter
    y_position = height - 40
    for line in final_summary.split("\n"):
        c.drawString(30, y_position, line)
        y_position -= 15
        if y_position < 40:
            c.showPage()
            y_position = height - 40
    c.save()
    pdf_bytes.seek(0)

    return final_summary, txt_bytes, pdf_bytes

# ✅ Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 📄 PDF Text Summarizer\nUpload a PDF and get an **AI-generated summary** using BART or PEGASUS.")

    model_choice = gr.Dropdown(
        ["facebook/bart-large-cnn", "google/pegasus-xsum"],
        label="Choose a model",
        value="facebook/bart-large-cnn"
    )

    pdf_input = gr.File(label="Upload your PDF", file_types=[".pdf"])
    summary_output = gr.Textbox(label="📜 Final Summary", lines=10)

    txt_download = gr.File(label="📥 Download Summary as TXT")
    pdf_download = gr.File(label="📥 Download Summary as PDF")

    submit_btn = gr.Button("Summarize PDF")

    # ✅ Connect function
    submit_btn.click(
        summarize_pdf,
        inputs=[pdf_input, model_choice],
        outputs=[summary_output, txt_download, pdf_download]
    )

# ✅ Hugging Face will auto-run this
if __name__ == "__main__":
    demo.launch()
