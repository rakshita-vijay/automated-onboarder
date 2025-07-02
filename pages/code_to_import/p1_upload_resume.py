# resume_input.py

import streamlit as st
import os
import git
from docx import Document
from pypdf import PdfReader
from streamlit_tree_select import tree_select
from styles import css_dark  # or css_light

from pages.code_to_import.p2_upload_jd import jd_button

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

def save_and_extract_resume(uploaded_file, applicant_name):
    base_dir = "scraped_info"
    applicant_dir = os.path.join(base_dir, applicant_name)
    os.makedirs(applicant_dir, exist_ok=True)
    original_path = os.path.join(applicant_dir, uploaded_file.name)
    with open(original_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    file_ext = uploaded_file.name.split(".")[-1].lower()
    if file_ext == "docx":
        text = extract_docx(original_path)
    elif file_ext == "pdf":
        text = extract_pdf(original_path)
    else:
        text = extract_txt(original_path)
    text_filename = f"{os.path.splitext(uploaded_file.name)[0]}.txt"
    text_path = os.path.join(applicant_dir, text_filename)
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(text)
    return text_path

def build_tree(root="scraped_info"):
    nodes = []
    if os.path.exists(root):
        for folder in sorted(os.listdir(root)):
            folder_path = os.path.join(root, folder)
            if os.path.isdir(folder_path):
                children = []
                for file in sorted(os.listdir(folder_path)):
                    file_path = os.path.join(folder_path, file)
                    if os.path.isfile(file_path):
                        children.append({"label": file, "value": f"{folder}/{file}"})
                nodes.append({"label": folder, "value": folder, "children": children})
    return nodes

def resume_button(): 
  st.page_link("pages/p1_resume.py", label="📄 Upload Resume")
  
def upload_resume():
    # Session state
    if 'repo' not in st.session_state:
        st.session_state.repo = None

    st.header("📄 Upload Resume & Supporting Documents")
    uploaded_files = st.file_uploader(
        "Upload files",
        label_visibility="collapsed",
        type=["docx", "pdf", "txt"],
        accept_multiple_files=True,
        help="Only .docx, .pdf, or .txt files are allowed.",
        key="file_uploader"
    )
    applicant_name = st.text_input(
        "👤 Applicant Name",
        placeholder="Enter applicant name for these files (e.g., Jane Doe)"
    )

    if st.button("Process Files"):
        if not uploaded_files:
            st.warning("There are no files to process :(")
        elif not applicant_name:
            st.warning("Applicant's name has not been entered :(")
        else:
            with st.spinner("Extracting and saving files..."):
                for uploaded_file in uploaded_files:
                    try:
                        save_and_extract_resume(uploaded_file, applicant_name.strip())
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                st.success(f"Successfully processed {len(uploaded_files)} files for {applicant_name}!")

    # GitHub integration
    st.divider()
    st.subheader("GitHub Integration")
    if st.button("Push to GitHub"):
        with st.spinner("Pushing to GitHub..."):
            try:
                if not st.session_state.repo:
                    st.session_state.repo = setup_git_repo()
                if st.session_state.repo:
                    repo = st.session_state.repo
                    repo.git.add("scraped_info/")
                    repo.index.commit("Add new applicant files")
                    origin = repo.remote(name="origin")
                    origin.push()
                    st.success("Files pushed to GitHub repository!")
            except Exception as e:
                st.error(f"Push failed: {str(e)}")

    # Display folder structure
    st.divider()
    st.subheader("📂 Uploaded Applicants's Resumes & Files")
    nodes = build_tree()
    if nodes:
        tree_select(nodes)
    else:
        st.info("No applicant files uploaded yet.")
    
    jd_button()
      
