import streamlit as st
from styles import css_dark

from pages.code_to_import.p1_upload_resume import resume_button 
from pages.code_to_import.p2_upload_jd import jd_button 
from pages.code_to_import.p3_crosscheck_resume_jd import resume_x_jd

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
  page_title="Evalute Resume & Supporting Documents",
  page_icon="🧮",
  layout="wide",
  initial_sidebar_state="expanded"
)

from the_sidebar import the_sb
the_sb()
    
resume_x_jd()
