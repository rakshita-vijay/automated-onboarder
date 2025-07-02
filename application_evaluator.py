import streamlit as st
import os
import shutil
import base64
import git
from docx import Document
from pypdf import PdfReader
import tempfile
import pandas as pd
from streamlit_tree_select import tree_select 

from resume_input import upload_resume
from jd_input import upload_jd

from styles import css_dark 
st.markdown(css_dark, unsafe_allow_html=True)  

# Streamlit app
def main(): 
    st.markdown(
        "<h1 style='text-align:center;'>📄✨ Document Processing System ✨📄</h1>",
        unsafe_allow_html=True
    ) 

    st.markdown("""
    <div style='text-align:center; font-size:1.2em;'>
        <span>🚀 <b>Upload files (DOCX, PDF, TXT)</b> to extract text content.</span><br>
        <span>🗂️ <b>Files will be processed and organized by applicant.</span><br>
        <span>📝 <b>Extracted text saved to structured folders.</span><br>
        <span>🌐 <b>Results pushed to <b>GitHub</b> repository.</span><br> <br> 
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'processed' not in st.session_state:
        st.session_state.processed = False
    if 'repo' not in st.session_state:
        st.session_state.repo = None 
        
    upload_resume()
    upload_jd()

if __name__ == "__main__":
    main()
