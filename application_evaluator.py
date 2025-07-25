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

# Add: Model startup calls
from pages.code_to_import.models.m1_1_completeness_model import CompletenessModel
from pages.code_to_import.models.m1_2_truthiness_model import TruthinessModel

# Run models if final CSV missing
if not os.path.exists("resume_final.csv"):
  CompletenessModel().run()
  TruthinessModel().run()

st.switch_page("pages/p0_home_page.py")
