import streamlit as st
import os
import git
from docx import Document
from pypdf import PdfReader
from streamlit_tree_select import tree_select
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

def crosscheck_button():
  st.page_link("pages/p3_eval.py", label="☯️ Evaluate Application") 

def resume_x_jd():
  pass



""" 
st.markdown(css_dark, unsafe_allow_html=True)
st.title("🧮 Evaluate Applications")
st.write("This page will allow you to compare resumes with job descriptions.")  
"""
