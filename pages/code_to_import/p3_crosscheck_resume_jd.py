import streamlit as st
import os
import git
from docx import Document
from pypdf import PdfReader
from streamlit_tree_select import tree_select
from styles import css_dark 

from pages.code_to_import.p1_upload_resume import resume_button
from pages.code_to_import.p2_upload_jd import jd_button

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
 
def list_txt_files_recursive_sorted(directory):
    """
    Recursively find all .txt files in all subfolders of 'directory', 
    and return a sorted list of their full paths (sorted alphabetically by filename).
    """
    txt_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt'): 
                rel_dir = os.path.relpath(root, directory)
                rel_file = os.path.join(rel_dir, file)
                txt_files.append(rel_file) 
              
    txt_files.sort(key=lambda x: os.path.basename(x).lower())
    return txt_files
  
def get_applicant_name_from_filename(filename):
    """Extract applicant name from resume filename (e.g., 'anupam.txt' -> 'anupam')."""
    return os.path.splitext(filename)[0]

def get_jd_name_from_filename(filename):
    """Extract JD name from JD filename (e.g., 'jd1.txt' -> 'jd1')."""
    return os.path.splitext(filename)[0]

def display_jd_content(jd_dir, jd_file):
    """Return the content of the selected JD file as a string."""
    jd_path = os.path.join(jd_dir, jd_file)
    if os.path.exists(jd_path):
        with open(jd_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "JD file not found."

def resume_x_jd():
    st.title("🧮 Evaluate Applications") 
    
    st.divider()
    resume_button()
    jd_button() 
    st.divider()

    scraped_info_dir = "scraped_info"
    jd_dir = "JDs"

    # List .txt files in both directories 
    resume_files = list_txt_files_recursive_sorted(scraped_info_dir)
    jd_files = list_txt_files_recursive_sorted(jd_dir)

    # Two columns for dropdowns
    col1, col2 = st.columns([1, 1])
    with col1:
        selected_resume = st.selectbox(
            "Select Resume File",
            resume_files,
            index=None,
            placeholder="Choose a resume...",
            key="resume_select"
        )
    with col2:
        selected_jd = st.selectbox(
            "Select JD File",
            jd_files,
            index=None,
            placeholder="Choose a JD...",
            key="jd_select"
        ) 

    # When both are selected, show the comparison box
    if selected_resume and selected_jd:
        applicant_name = get_applicant_name_from_filename(selected_resume)
        jd_name = get_jd_name_from_filename(selected_jd)
        jd_content = display_jd_content(jd_dir, selected_jd)

        # Display box with 3 columns: applicant, X, JD
        box_col1, box_col2, box_col3 = st.columns([2, 1, 5])
        with box_col1:
            st.markdown(f"<div style='padding: 1em; border: 1px solid #888; border-radius: 8px; background: #222; color: #fff; text-align: center; font-weight: bold;'>{applicant_name}</div>", unsafe_allow_html=True)
        with box_col2:
            st.markdown(f"<div style='padding: 1em; border: 1px solid #888; border-radius: 8px; background: #333; color: #fff; text-align: center; font-size: 1.5em;'>×</div>", unsafe_allow_html=True)
        with box_col3:
            st.markdown(f"<div style='padding: 1em; border: 1px solid #888; border-radius: 8px; background: #222; color: #fff; text-align: center; font-weight: bold;'>{jd_name}</div>", unsafe_allow_html=True)

        # Optionally: show JD content below
        st.markdown("#### Job Description Content")
        st.code(jd_content, language="text")
    else:
        st.info("Please select both a resume and a JD file to compare.") 
