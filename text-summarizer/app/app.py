import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pdf_utils import extract_text_from_pdf, chunk_text
# from src.summarizer import summarize_text
# (rest of your imports)

import streamlit as st
# from src.pdf_utils import extract_text_from_pdf, chunk_text
from src.summarizer import Summarizer
from io import BytesIO
from reportlab.pdfgen import canvas   # for PDF download
from reportlab.lib.pagesizes import letter

st.set_page_config(page_title="PDF Summarizer", page_icon="📄")

st.title(" PDF Text Summarizer")
st.markdown("Upload a PDF and get an **AI-generated summary** using BART or PEGASUS.")

# Model selection
model_choice = st.selectbox("Choose a model:", ["facebook/bart-large-cnn", "google/pegasus-xsum"])
summarizer = Summarizer(model_choice)

#  File uploader
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Extract text
    text = extract_text_from_pdf("temp.pdf")
    st.success(" PDF successfully loaded and text extracted.")

    # Summarize text (handle long PDFs)
    chunks = chunk_text(text)
    st.write(f" The PDF is split into {len(chunks)} chunk(s) for processing.")
    
    summaries = []
    for idx, chunk in enumerate(chunks, start=1):
        st.write(f"⏳ Summarizing chunk {idx}...")
        summary = summarizer.summarize(chunk)
        summaries.append(summary)

    final_summary = " ".join(summaries)

    st.subheader("📜 Final Summary")
    st.write(final_summary)

    #  DOWNLOAD AS TXT
    txt_bytes = BytesIO()
    txt_bytes.write(final_summary.encode('utf-8'))
    txt_bytes.seek(0)

    st.download_button(
        label="📥 Download Summary as TXT",
        data=txt_bytes,
        file_name="summary.txt",
        mime="text/plain"
    )

    # DOWNLOAD AS PDF
    pdf_bytes = BytesIO()
    c = canvas.Canvas(pdf_bytes, pagesize=letter)
    width, height = letter

    # Write summary into PDF (basic layout)
    y_position = height - 40
    for line in final_summary.split("\n"):
        c.drawString(30, y_position, line)
        y_position -= 15
        if y_position < 40:
            c.showPage()
            y_position = height - 40

    c.save()
    pdf_bytes.seek(0)

    st.download_button(
        label="Download Summary as PDF",
        data=pdf_bytes,
        file_name="summary.pdf",
        mime="application/pdf"
    )
