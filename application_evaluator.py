import streamlit as st
import os
import shutil
import base64
import git
from docx import Document
from pypdf import PdfReader
import tempfile

from styles import css 
st.markdown(css, unsafe_allow_html=True)

def setup_git_repo():
    try:
        # Initialize git repo in current directory (if not already)
        GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]
        repo = git.Repo(".")  # Use current directory, not scraped_info
        username = "rakshita-vijay"
        repo_url = f"https://{username}:{GITHUB_TOKEN}@github.com/{username}/automated-onboarder.git" 
        repo.remote().set_url(repo_url)
        st.success("Using existing Git repository!")
        return repo
    except git.exc.InvalidGitRepositoryError:
        st.error("Not in a Git repository. Make sure you're running from your repo directory.")
        return None 

# File processing functions
def extract_docx(file_path):
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_pdf(file_path):
    with open(file_path, "rb") as f:
        reader = PdfReader(f)
        return "\n".join([page.extract_text() for page in reader.pages])

def extract_txt(file_path):
    with open(file_path, "r") as f:
        return f.read()

# Streamlit app
def main(): 
    st.markdown(
        "<h1 style='text-align:center;'>📄✨ Document Processing System ✨📄</h1>",
        unsafe_allow_html=True
    ) 

    st.markdown("""
    <div style='text-align:center; font-size:1.2em;'>
        <span>🚀 <b>Upload files (DOCX, PDF, TXT)</b> to extract text content.</span><br>
        <ul style="list-style: none; padding-left: 50px; text-align: left;">
            <li>🗂️ Files will be processed and organized by applicant</li>
            <li>📝 Extracted text saved to structured folders</li>
            <li>🌐 Results pushed to <b>GitHub</b> repository</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'processed' not in st.session_state:
        st.session_state.processed = False
    if 'repo' not in st.session_state:
        st.session_state.repo = None
    
    # File uploader 
    uploaded_files = st.file_uploader(
        "📤 <b>Drag and drop files here</b> (Limit 400MB per file, DOCX, PDF, TXT)",
        type=["docx", "pdf", "txt"],
        accept_multiple_files=True,
        help="Only .docx, .pdf, or .txt files are allowed.",
        key="file_uploader"
    )
    
    # Applicant name input
    applicant_name = st.text_input(
        "👤 Applicant Name",
        placeholder="Enter applicant name for these files"
    )

    if st.button("Process Files"):
        if not uploaded_files:
            st.warning("There are no files to process")
        elif not applicant_name:
            st.warning("Applicant's name has not been entered")
        else:
            with st.spinner("Processing files..."):
                # Setup folder structure within current repo
                base_dir = "scraped_info"
                applicant_dir = os.path.join(base_dir, applicant_name)
                os.makedirs(applicant_dir, exist_ok=True)
                
                # Process each file (same as before)
                for uploaded_file in uploaded_files:
                    # Save original file
                    original_path = os.path.join(applicant_dir, uploaded_file.name)
                    with open(original_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Extract text and save (same as before)
                    file_ext = uploaded_file.name.split(".")[-1].lower()
                    try:
                        if file_ext == "docx":
                            text = extract_docx(original_path)
                        elif file_ext == "pdf":
                            text = extract_pdf(original_path)
                        else:
                            text = extract_txt(original_path)
                        
                        text_filename = f"{os.path.splitext(uploaded_file.name)[0]}.txt"
                        text_path = os.path.join(applicant_dir, text_filename)
                        with open(text_path, "w") as f:
                            f.write(text)
                            
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                
                st.session_state.processed = True
                st.success(f"Processed {len(uploaded_files)} files for {applicant_name}!") 
    
    # GitHub integration
    if st.session_state.processed:
        st.divider()
        st.subheader("GitHub Integration")

        if st.button("Push to GitHub"):
            with st.spinner("Pushing to GitHub..."):
                try:
                    # Initialize repo if not already set up
                    if not st.session_state.repo:
                        st.session_state.repo = setup_git_repo()
                    
                    if st.session_state.repo:
                        repo = st.session_state.repo
                        repo.git.add("scraped_info/")  # Add only the scraped_info folder
                        repo.index.commit("Add new applicant files")
                        origin = repo.remote(name="origin")
                        origin.push()
                        st.success("Files pushed to GitHub repository!")
                except Exception as e:
                    st.error(f"Push failed: {str(e)}") 
        
        # Display folder structure
        st.subheader("Folder Structure")
        if os.path.exists("scraped_info"):
            st.markdown("<h3 style='margin-top:2em;'>📁 <u>Folder Structure</u></h3>", unsafe_allow_html=True)
            for applicant in os.listdir("scraped_info"):
                applicant_path = os.path.join("scraped_info", applicant)
                if os.path.isdir(applicant_path):  # Only process directories 
                    st.markdown(f"<div class='folder-box'><b>👤 {applicant}</b>", unsafe_allow_html=True)
                    for file in os.listdir(applicant_path):
                        st.markdown(f"<span style='color:#6c63ff;'>🗎 {file}</span>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                else:
                    # Optionally, display or log non-directory items
                    pass  # or st.write(f"Skipping non-directory: {applicant}") 
        
        # Download ZIP option
        st.divider()
        st.subheader("Download Results")
        if os.path.exists("scraped_info"):
            shutil.make_archive("scraped_info", 'zip', "scraped_info")
            with open("scraped_info.zip", "rb") as f:
                st.download_button(
                    label="Download All Files as ZIP",
                    data=f,
                    file_name="scraped_info.zip"
                )

if __name__ == "__main__":
    main()
