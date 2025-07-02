import streamlit as st
from code_to_import.p1_upload_resume import upload_resume
from styles import css_dark

st.markdown(css_dark, unsafe_allow_html=True)
upload_resume() 
