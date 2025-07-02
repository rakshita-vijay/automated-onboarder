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

# from pages.p1_upload_resume import upload_resume
# from pages.p2_upload_jd import upload_jd 

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
        
    st.divider()
    st.markdown("### What would you like to do?", unsafe_allow_html=True)
    # st.page_link("pages/p1_resume.py", label="Upload Resume", icon="📄")
    # st.page_link("pages/p2_jd.py", label="Upload JD", icon="📝")
    # st.page_link("pages/p3_eval.py", label="Evaluate Applications", icon="🧮") 
    # st.markdown('<div class="outline-btn">📄 Upload Resume</div>', unsafe_allow_html=True)
    # st.markdown('<div class="outline-btn">📝 Upload JD</div>', unsafe_allow_html=True)
    # st.markdown('<div class="outline-btn">🧮 Evaluate Applications</div>', unsafe_allow_html=True) 

    st.markdown("""
    <a href="/upload-resume/" style="text-decoration:none;">
        <div class="outline-btn">📄 Upload Resume</div>
    </a>
    <a href="/upload-jd/" style="text-decoration:none;">
        <div class="outline-btn">📝 Upload JD</div>
    </a>
    <a href="/evaluation/" style="text-decoration:none;">
        <div class="outline-btn">🧮 Evaluate Applications</div>
    </a>
    """, unsafe_allow_html=True)

    
    # upload_resume()
    # upload_jd()

if __name__ == "__main__":
    main()
