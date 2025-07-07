import streamlit as st

st.markdown("""
<style>
/* Remove all Streamlit sidebar styling for a flat, simple look */
[data-testid="stSidebar"] {
    background: none !important;
    box-shadow: none !important;
    border: none !important;
    padding: 0 !important;
}
[data-testid="stSidebarNav"] {
    background: none !important;
    box-shadow: none !important;
    border: none !important;
    padding: 0 !important;
}
[data-testid="stSidebarHeader"] {
    display: none !important;
}
[data-testid="stSidebarUserContent"] {
    padding: 0 !important;
}
[data-testid="stSidebar"] * {
    background: none !important;
    box-shadow: none !important;
    border: none !important;
}
</style>
""", unsafe_allow_html=True)

def the_sb():
  with st.sidebar:  
    st.page_link("pages/p0_home_page.py", label="🖋️ Application Evaluator")
    st.page_link("pages/p1_resume.py", label="📝 Upload Resume")
    st.page_link("pages/p2_jd.py", label="👔 Upload JDs") 
    st.page_link("pages/p3_eval.py", label="🧮 Evalute Uploaded Documents")
