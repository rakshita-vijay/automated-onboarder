import streamlit as st
import os
import shutil
import base64
import git
from docx import Document
from pypdf import PdfReader
import tempfile
import pandas as pd
from streamlit_tree_select import tree_select

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

def build_tree(root):
    nodes = []
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
    
# Streamlit app
def main(): 
    st.markdown(
        "<h1 style='text-align:center;'>📄✨ Document Processing System ✨📄</h1>",
        unsafe_allow_html=True
    ) 

    st.markdown("""
    <div style='text-align:center; font-size:1.2em;'>
        <span>🚀 <b>Upload files (DOCX, PDF, TXT)</b> to extract text content.</span><br>
        <span>🗂️ <b>Files will be processed and organized by applicant.</span><br>
        <span>📝 <b>Extracted text saved to structured folders.</span><br>
        <span>🌐 <b>Results pushed to <b>GitHub</b> repository.</span><br> <br> 
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'processed' not in st.session_state:
        st.session_state.processed = False
    if 'repo' not in st.session_state:
        st.session_state.repo = None
    
    # File uploader 
    uploaded_files = st.file_uploader(
        "Upload files",
        label_visibility="collapsed",
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
        scraped_info_path = "scraped_info"
        if os.path.exists(scraped_info_path):
            tree_data = build_tree(scraped_info_path)
            selected = tree_select(tree_data, checkbox=False, expand_all=True)
            # Display the selected file/folder if needed
            if selected:
                st.write("Selected:", selected)
            # st.markdown("<h3 style='margin-top:2em;'>📁 <u>Folder Structure</u></h3>", unsafe_allow_html=True)
            # files_data = []
            # for applicant in os.listdir(scraped_info_path):
            #     applicant_path = os.path.join(scraped_info_path, applicant)
            #     if os.path.isdir(applicant_path):  # Only process directories 
            #         st.markdown(f"<div class='folder-box'><b>👤 {applicant}</b>", unsafe_allow_html=True)
            #         for file in os.listdir(applicant_path):
            #             st.markdown(f"<span style='color:#6c63ff;'>🗎 {file}</span>", unsafe_allow_html=True)
            #             files_data.append({"Applicant": applicant, "File": file})

            # if files_data:
            #     df = pd.DataFrame(files_data)
                
            #     # Style the dataframe for dark mode (light background)
            #     styled_df = df.style.set_properties(**{
            #         'background-color': '#f0f9ff',  # Light baby blue
            #         'color': '#222',                # Dark text
            #         'border-color': '#a5b4fc'
            #     }).set_table_styles([
            #         {'selector': 'th', 'props': [
            #             ('background-color', '#e0e7ff'),
            #             ('color', '#222'),
            #             ('font-weight', 'bold')
            #         ]}
            #     ])
                
            #     st.dataframe(styled_df, use_container_width=True)
        
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
