import streamlit as st

st.title("📄 Document Uploader")

st.write("Upload one or more files (DOCX, PDF, or TXT) by dragging and dropping or using the 'Browse files' button below.")

uploaded_files = st.file_uploader( 
    type=["docx", "pdf", "txt"],
    accept_multiple_files=True,
    help="Only .docx, .pdf, or .txt files are allowed."
)

if uploaded_files is not None:
    st.write("### Uploaded File Details:")
    for uploaded_file in uploaded_files:
        file_details = {
            "Filename": uploaded_file.name,
            "File Type": uploaded_file.type,
            "File Size (bytes)": uploaded_file.size
        }
        st.json(file_details)
else:
    st.info("Awaiting file upload. You can drag and drop or click to select files.")
