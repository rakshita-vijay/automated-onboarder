import streamlit as st
from styles import css_dark

from pages.code_to_import.p2_upload_jd import upload_jd 
from pages.code_to_import.p1_upload_resume import resume_button 

st.markdown(css_dark, unsafe_allow_html=True)
upload_jd() 
st.divider()
resume_button() 
