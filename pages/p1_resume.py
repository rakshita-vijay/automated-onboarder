import streamlit as st
from styles import css_dark

from pages.code_to_import.p1_upload_resume import upload_resume
from pages.code_to_import.p2_upload_jd import jd_button 

st.markdown(css_dark, unsafe_allow_html=True)
upload_resume() 
st.divider()
jd_button()
