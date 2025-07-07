import streamlit as st 

def the_sb():
  with st.sidebar:  
    st.page_link("pages/p0_home_page.py", label="🖋️ Evaluator Home Page")
    st.page_link("pages/p1_resume.py", label="📝 Upload Resume")
    st.page_link("pages/p2_jd.py", label="👔 Upload JDs") 
    st.page_link("pages/p3_eval.py", label="🧮 Evalute Uploaded Documents")
