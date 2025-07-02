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

col1, col2 = st.columns(2)

 with col1:
    resume_button()
with col2:
    jd_button()

# resume_button()
# jd_button()
st.divider()
resume_x_jd()
