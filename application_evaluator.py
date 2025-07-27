import streamlit as st
import os, sys, shutil, base64, git
from docx import Document
from pypdf import PdfReader
import tempfile
import pandas as pd
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

if 'app_rebooted' not in st.session_state:
  st.session_state.app_rebooted = True

# Model startup calls
try:
  from pages.code_to_import.models.m1_1_completeness_model import CompletenessModel
  from pages.code_to_import.models.m1_2_truthiness_model import TruthinessModel

  if 'app_rebooted' not in st.session_state:
    st.session_state.app_rebooted = True

  TRAIN_RESUMES_DIR = "training_data/training_resumes"
  os.makedirs(TRAIN_RESUMES_DIR, exist_ok=True)

  TRAIN_JDS_DIR = "training_data/training_jds"
  os.makedirs(TRAIN_JDS_DIR, exist_ok=True)

  from pages.code_to_import.models.m1_1_completeness_model import create_initial_dataset
  create_initial_dataset()

  if not os.path.exists("resume_final.csv") or st.session_state.app_rebooted:
    with st.spinner("Initializing AI models..."):
      CompletenessModel().run()
      TruthinessModel().run()

    st.session_state.app_rebooted = False
except ImportError as e:
  st.warning(f"Model import failed: {e}. Continuing without AI features.")

st.switch_page("pages/p0_home_page.py")
