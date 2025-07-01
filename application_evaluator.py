import streamlit as st
import os
import shutil
import base64
import git
from docx import Document
from pypdf import PdfReader
import tempfile

# Initialize GitHub repository
def setup_git_repo():
    repo_url = "https://github.com/yourusername/your-repo.git"  # Replace with your repo
    repo_path = "scraped_info"
    
    try:
        if not os.path.exists(repo_path):
            repo = git.Repo.clone_from(repo_url, repo_path)
            st.success("Cloned GitHub repository successfully!")
        else:
            repo = git.Repo(repo_path)
            repo.remotes.origin.pull()
            st.success("Updated existing repository!")
        return repo
    except git.exc.GitCommandError as e:
        st.error(f"GitHub error: {str(e)}")
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
    st.title("📄 Document Processing System")
    st.markdown("""
    **Upload files (DOCX, PDF, TXT) to extract text content.**
    - Files will be processed and organized by applicant
    - Extracted text saved to structured folders
    - Results pushed to GitHub repository
    """)
    
    # Initialize session state
    if 'processed' not in st.session_state:
        st.session_state.processed = False
    if 'repo' not in st.session_state:
        st.session_state.repo = None
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Drag and drop files here",
        type=["docx", "pdf", "txt"],
        accept_multiple_files=True
    )
    
    # Applicant name input
    applicant_name = st.text_input("Applicant Name", 
                                 placeholder="Enter applicant name for these files")
    
    if st.button("Process Files") and uploaded_files and applicant_name:
        with st.spinner("Processing files..."):
            # Setup folder structure
            base_dir = "scraped_info"
            applicant_dir = os.path.join(base_dir, applicant_name)
            os.makedirs(applicant_dir, exist_ok=True)
            
            # Process each file
            for uploaded_file in uploaded_files:
                # Save original file
                original_path = os.path.join(applicant_dir, uploaded_file.name)
                with open(original_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Extract text based on file type
                file_ext = uploaded_file.name.split(".")[-1].lower()
                try:
                    if file_ext == "docx":
                        text = extract_docx(original_path)
                    elif file_ext == "pdf":
                        text = extract_pdf(original_path)
                    else:  # txt
                        text = extract_txt(original_path)
                    
                    # Save extracted text
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
                        repo.git.add("--all")
                        repo.index.commit("Add processed files")
                        origin = repo.remote(name="origin")
                        origin.push()
                        st.success("Files pushed to GitHub repository!")
                except Exception as e:
                    st.error(f"Push failed: {str(e)}")
        
        # Display folder structure
        st.subheader("Folder Structure")
        if os.path.exists("scraped_info"):
            for applicant in os.listdir("scraped_info"):
                st.markdown(f"### 📁 {applicant}")
                applicant_path = os.path.join("scraped_info", applicant)
                for file in os.listdir(applicant_path):
                    st.write(f"- {file}")
        
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
