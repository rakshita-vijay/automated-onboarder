import streamlit as st

st.title("📄 Document Uploader")

st.write("Upload a file (DOCX, PDF, or TXT) by dragging and dropping or using the 'Browse files' button below.")

uploaded_file = None
uploaded_file = st.file_uploader(
    "Choose a file",
    type=["docx", "pdf", "txt"],
    help="Only .docx, .pdf, or .txt files are allowed."
)

if uploaded_file is not None:
    file_details = {
        "Filename": uploaded_file.name,
        "File Type": uploaded_file.type,
        "File Size (bytes)": uploaded_file.size
    }
    st.write("### File details:")
    st.json(file_details)
else:
    st.info("Awaiting file upload. You can drag and drop or click to select a file.")
