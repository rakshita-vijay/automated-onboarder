import streamlit as st
import os
from docx import Document
from pypdf import PdfReader
from streamlit_tree_select import tree_select

from pages.code_to_import.p1_upload_resume import resume_button
from pages.code_to_import.p2_upload_jd import jd_button

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
  st.page_link("pages/p3_eval.py", label="🧮 Evaluate Application")

def list_txt_files_recursive_sorted(directory):
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
  return os.path.splitext(os.path.basename(filename))[0].split('_resume')[0]

def display_jd_content(jd_dir, jd_file):
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

  resume_files_full = list_txt_files_recursive_sorted(resume_and_supporting_docs_dir)
  jd_files_full = list_txt_files_recursive_sorted(jd_dir)

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

  if st.button("Check selected File against JD"):
    if selected_resume_name and selected_jd_name:
      with st.spinner("Check selected File against JD..."):
        selected_resume = resume_file_map[selected_resume_name]
        selected_jd = jd_file_map[selected_jd_name]
        applicant_name = get_applicant_name_from_filename(selected_resume)
        jd_name = os.path.splitext(selected_jd)[0]

        # Read resume text
        resume_path = os.path.join(resume_and_supporting_docs_dir, selected_resume)
        with open(resume_path, 'r', encoding='utf-8') as f:
          resume_text = f.read()

        jd_content = display_jd_content(jd_dir, selected_jd)

        # Call the model
        from pages.code_to_import.models.m1_0_crosschecker_model import CrossCheckerModel
        try:
          crosschecker = CrossCheckerModel()
          scores = crosschecker.get_scores(applicant_name, resume_text, jd_content)

          # Display scores
          st.subheader("📊 Evaluation Scores")
          col1, col2, col3 = st.columns(3)
          with col1:
            st.metric("Completeness", f"{scores['completeness']:.1f}%")
          with col2:
            st.metric("Truthiness", f"{scores['truthiness']:.1f}%")
          with col3:
            st.metric("Relevance", f"{scores['relevance']:.1f}%")
        except Exception as e:
          st.error(f"Model evaluation failed: {str(e)}")

        # Display comparison box
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
