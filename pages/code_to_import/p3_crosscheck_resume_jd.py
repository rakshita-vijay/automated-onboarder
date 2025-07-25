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
st.markdown(""" """, unsafe_allow_html=True)
def crosscheck_button():
  st.page_link("pages/p3_eval.py", label="🧮 Evaluate Application")
def list_txt_files_recursive_sorted(directory):
  """ Recursively find all .txt files in all subfolders of 'directory', and return a sorted list of their full paths (sorted alphabetically by filename). """
  txt_files = []
  for root, dirs, files in os.walk(directory):
    for file in files:
      if file.endswith('.txt'):
        if directory != "JDs":
          rel_dir = os.path.relpath(root, directory)
          rel_file = os.path.join(rel_dir, file)
          txt_files.append(rel_file)
        else:
          txt_files.append(file)
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
  resume_and_supporting_docs_dir = "resume_and_supporting_docs"
  jd_dir = "JDs"
  # (existing code up to line 77)
  # Add: After user selection, call the model
  from pages.code_to_import.models.m1_0_crosschecker_model import CrossCheckerModel
  crosschecker = CrossCheckerModel()
  # Inside the if selected_resume_name and selected_jd_name block (after line 77)
  resume_path = os.path.join(resume_and_supporting_docs_dir, resume_file_map[selected_resume_name])  # Actual path
  with open(resume_path, 'r', encoding='utf-8') as f:
    resume_text = f.read()
  jd_text = display_jd_content(jd_dir, selected_jd)  # Already there
  applicant_name = get_applicant_name_from_filename(selected_resume)  # Already there
  scores = crosschecker.get_scores(applicant_name, resume_text, jd_text)
  st.write("**Completeness:**", scores["completeness"])
  st.write("**Truthiness:**", scores["truthiness"])
  st.write("**Relevance:**", scores["relevance"])
  # (rest of file)
  # New code: show only file names, but keep mapping to full path
  resume_files_full = list_txt_files_recursive_sorted(resume_and_supporting_docs_dir)
  jd_files_full = list_txt_files_recursive_sorted(jd_dir)
  # Build mapping: filename -> full relative path
  resume_file_map = {os.path.basename(f): f for f in resume_files_full}
  jd_file_map = {os.path.basename(f): f for f in jd_files_full}
  col1, col2 = st.columns([1, 1])
  with col1:
    selected_resume_name = st.selectbox(
      "Select Resume File",
      list(resume_file_map.keys()),
      index=None,
      placeholder="Choose a resume...",
      key="resume_select"
    )
  with col2:
    selected_jd_name = st.selectbox(
      "Select JD File",
      list(jd_file_map.keys()),
      index=None,
      placeholder="Choose a JD...",
      key="jd_select"
    )
  # Use the mapping to get the full path when needed
  if selected_resume_name and selected_jd_name:
    selected_resume = resume_file_map[selected_resume_name]
    selected_jd = jd_file_map[selected_jd_name]
    applicant_name = get_applicant_name_from_filename(selected_resume)
    jd_name = get_jd_name_from_filename(selected_jd)
    jd_content = display_jd_content(jd_dir, selected_jd)
    # Display box with 3 columns: applicant, X, JD
    box_col1, box_col2, box_col3 = st.columns([5, 1, 5])
    with box_col1:
      st.markdown(f"<div style='padding: 1em; border: 1px solid #888; border-radius: 8px; background: #222; color: #fff; text-align: center; font-weight: bold;'>{applicant_name}</div>", unsafe_allow_html=True)
    with box_col2:
      st.markdown(f"<div style='padding: 1em; border: 1px solid #888; border-radius: 8px; background: #333; color: #fff; text-align: center; font-size: 1.5em;'>×</div>", unsafe_allow_html=True)
    with box_col3:
      st.markdown(f"<div style='padding: 1em; border: 1px solid #888; border-radius: 8px; background: #222; color: #fff; text-align: center; font-weight: bold;'>{jd_name}</div>", unsafe_allow_html=True)
    st.markdown("#### Job Description Content")
    st.code(jd_content, language="text")
  else:
    st.info("Please select both a resume and a JD file to compare.")
