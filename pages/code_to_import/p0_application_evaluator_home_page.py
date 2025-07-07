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

from pages.code_to_import.p1_upload_resume import resume_button
from pages.code_to_import.p2_upload_jd import jd_button
from pages.code_to_import.p3_crosscheck_resume_jd import crosscheck_button 

from styles import css_dark  

st.markdown(css_dark, unsafe_allow_html=True)
st.markdown("""
<style> 
@media (prefers-color-scheme: dark) {
  [data-testid="stTreeSelect"] span,
  [data-testid="stTreeSelect"] label,
  [data-testid="stTreeSelect"] div {
    color: #fff !important;
  }
  [data-testid="stTreeSelect"] input[type="checkbox"] {
    filter: invert(1); /* Optional: makes the checkbox white */
  }
} 
</style>
""", unsafe_allow_html=True)

st.set_page_config(
  page_title="Application Evaluator",
  page_icon="🖋️",
  layout="wide",
  initial_sidebar_state="expanded"
)

with st.sidebar: 
  st.page_link("pages/code_to_import/p0_application_evaluator_home_page.py", label="Application Evaluator", icon="🖋️")
  st.page_link("pages/code_to_import/p1_resume.py", label="Upload Resume & Supporting Documents", icon="📝")
  st.page_link("pages/code_to_import/p2_jd.py", label="Upload Job Description", icon="👔") 
  st.page_link("pages/code_to_import/p3_eval.py", label="Evalute Resume & Supporting Documents", icon="🧮")
 
# Streamlit app
def app_eval_home_page(): 
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
    resume_button()
    jd_button()
    crosscheck_button()  
