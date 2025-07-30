import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from src.pdf_utils import extract_text_from_pdf, chunk_text
from src.summarizer import Summarizer
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

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


# 🚀 STREAMLIT APP
st.title("📄 PDF Text Summarizer")
st.write("Upload a PDF and get an **AI-generated summary** using BART or PEGASUS.")

# Model selection
model_choice = st.selectbox(
    "Choose a model:",
    ["facebook/bart-large-cnn", "google/pegasus-xsum"]
)

# PDF upload
pdf_file = st.file_uploader("📂 Upload your PDF", type=["pdf"])

if pdf_file:
    if st.button("Summarize PDF"):
        with st.spinner("Summarizing... Please wait ⏳"):
            summary, txt_bytes, pdf_bytes = summarize_pdf(pdf_file, model_choice)

        # ✅ Display summary
        st.subheader("📜 Final Summary")
        st.text_area("", summary, height=300)

        # ✅ Provide download buttons
        st.download_button(
            label="📥 Download Summary as TXT",
            data=txt_bytes,
            file_name="summary.txt",
            mime="text/plain"
        )

        st.download_button(
            label="📥 Download Summary as PDF",
            data=pdf_bytes,
            file_name="summary.pdf",
            mime="application/pdf"
        )
