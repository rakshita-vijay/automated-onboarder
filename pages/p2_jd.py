import streamlit as st
from code_to_import.p2_upload_jd import upload_jd
from styles import css_dark

st.markdown(css_dark, unsafe_allow_html=True)
upload_jd() 
