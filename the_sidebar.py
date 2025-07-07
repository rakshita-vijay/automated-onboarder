import streamlit as st

def the_sb():
  with st.sidebar: 
    st.page_link("pages/p0_home_page.py", label="Application Evaluator", icon="🖋️")
    st.page_link("pages/p1_resume.py", label="Upload Resume & Supporting Documents", icon="📝")
    st.page_link("pages/p2_jd.py", label="Upload Job Description", icon="👔") 
    st.page_link("pages/p3_eval.py", label="Evalute Resume & Supporting Documents", icon="🧮")
