import streamlit as st
from code_to_import.jd_input import upload_jd
from styles import css_dark

st.markdown(css_dark, unsafe_allow_html=True)
upload_jd() 
