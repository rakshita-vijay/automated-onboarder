import streamlit as st
from styles import css_dark

from pages.code_to_import.p1_upload_resume import upload_resume
from pages.code_to_import.p2_upload_jd import jd_button  
from pages.code_to_import.p3_crosscheck_resume_jd import crosscheck_button 

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
  page_title="Upload Resume & Supporting Documents",
  page_icon="📝",
  layout="wide",
  initial_sidebar_state="expanded"
)

with st.sidebar: 
  st.page_link("pages/p0_home_page.py", label="Application Evaluator", icon="🖋️")
  st.page_link("pages/p1_resume.py", label="Upload Resume & Supporting Documents", icon="📝")
  st.page_link("pages/p2_jd.py", label="Upload Job Description", icon="👔") 
  st.page_link("pages/p3_eval.py", label="Evalute Resume & Supporting Documents", icon="🧮")
    
upload_resume() 
st.divider()
jd_button()
crosscheck_button() 
