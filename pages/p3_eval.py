import streamlit as st
from pages.code_to_import.p3_crosscheck_resume_jd import resume_x_jd
from styles import css_dark

st.markdown(css_dark, unsafe_allow_html=True)
resume_x_jd() 
