# jd_input.py

import streamlit as st
import os
import git
from docx import Document
from pypdf import PdfReader
from streamlit_tree_select import tree_select
from styles import css_dark  # or css_light, as appropriate

from code_to_import.p1_upload_resume import resume_button 

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

def setup_git_repo():
    try:
        GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]
        repo = git.Repo(".")
        username = "rakshita-vijay"
        repo_url = f"https://{username}:{GITHUB_TOKEN}@github.com/{username}/automated-onboarder.git"
        repo.remote().set_url(repo_url)
        st.success("Using existing Git repository!")
        return repo
    except git.exc.InvalidGitRepositoryError:
        st.error("Not in a Git repository. Make sure you're running from your repo directory.")
        return None

def extract_docx(file_path):
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_pdf(file_path):
    with open(file_path, "rb") as f:
        reader = PdfReader(f)
        return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

def extract_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def save_and_extract_jd(jd_file, jd_name):
    jd_dir = "JDs"
    os.makedirs(jd_dir, exist_ok=True)
    ext = jd_file.name.split(".")[-1].lower()
    jd_path = os.path.join(jd_dir, f"{jd_name}.{ext}")
    with open(jd_path, "wb") as f:
        f.write(jd_file.getbuffer())
    # Extract text
    if ext == "docx":
        text = extract_docx(jd_path)
    elif ext == "pdf":
        text = extract_pdf(jd_path)
    else:
        text = extract_txt(jd_path)
    txt_path = os.path.join(jd_dir, f"{jd_name}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    # return txt_path

def build_jd_tree(jd_dir="JDs"):
    nodes = []
    if os.path.exists(jd_dir):
        for file in sorted(os.listdir(jd_dir)):
            if file.endswith(".txt"):
                nodes.append({"label": file[:-4], "value": file})
    return nodes

def jd_button():
  st.page_link("pages/p2_jd.py", label="📝 Upload JD") 
  
def upload_jd():
    # Session state for Git
    if 'jd_repo' not in st.session_state:
        st.session_state.jd_repo = None

    st.header("📄 Upload a Job Description (JD)")
    jd_file = st.file_uploader(
        "Upload a JD file",
        label_visibility="collapsed",
        type=["docx", "pdf", "txt"],
        key="jd_uploader",
        help="Upload a single JD file (DOCX, PDF, or TXT)."
    )
    jd_name = st.text_input(
        "📝 JD Name",
        placeholder="Enter a name for this JD (e.g., Web Developer)",
        key="jd_name"
    )

    if st.button("Process JD"):
        if jd_file is None:
            st.warning("There are no JDs to process :(")
        elif not jd_name.strip():
            st.warning("JD name has not been entered :(")
        else:
            with st.spinner("Extracting and saving JD..."):   
                try:
                    save_and_extract_jd(jd_file, jd_name.strip())
                except Exception as e:
                    st.error(f"Error processing {jd_file.name}: {str(e)}")  
                st.success(f"JD '{jd_name}' extracted and saved as {jd_file.name}!")

    # GitHub integration for JD folder
    st.divider()
    st.subheader("GitHub Integration")
    if st.button("Push JDs to GitHub"):
        with st.spinner("Pushing JDs to GitHub..."):
            try:
                if not st.session_state.jd_repo:
                    st.session_state.jd_repo = setup_git_repo()
                if st.session_state.jd_repo:
                    repo = st.session_state.jd_repo
                    repo.git.add("JDs/")
                    repo.index.commit("Add or update JD files")
                    origin = repo.remote(name="origin")
                    origin.push()
                    st.success("JDs pushed to GitHub repository!")
            except Exception as e:
                st.error(f"Push failed: {str(e)}")

    st.divider()
    st.subheader("📂 Available Job Descriptions")
    nodes = build_jd_tree()
    if nodes:
        tree_select(nodes)
    else:
        st.info("No JDs uploaded yet.")

    resume_button()
