import streamlit as st
import os
import git
from docx import Document
from pypdf import PdfReader
from streamlit_tree_select import tree_select
from styles import css_dark  # or css_light

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

# --- GIT SETUP (unchanged) ---
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

def resume_button():
  st.page_link("pages/p1_resume.py", label="📄 Upload Resume")

# --- FILE EXTRACTION HELPERS (unchanged) ---
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

# --- SUPPORTING DOC NUMBERING (CHANGED) ---
def get_next_supporting_number(applicant_dir, name_prefix):
  """Return the next supporting doc number for this applicant."""
  prefix = f"{name_prefix}_supporting_"
  existing = [
    fname for fname in os.listdir(applicant_dir)
    if fname.startswith(prefix) and '.' in fname
  ]
  nums = []
  for fname in existing:
    try:
      num = int(fname.replace(prefix, '').split('.')[0])
      nums.append(num)
    except Exception:
      continue
  if nums:
    return max(nums) + 1
  else:
    return 1

# --- FILE SAVING AND EXTRACTION ---
def save_and_extract_file(uploaded_file, applicant_dir, save_name):
  """Save file and extract text to .txt in the same folder."""
  os.makedirs(applicant_dir, exist_ok=True)
  file_ext = uploaded_file.name.split(".")[-1].lower()
  file_path = os.path.join(applicant_dir, save_name)
  # Save original file
  with open(file_path, "wb") as f:
    f.write(uploaded_file.getbuffer())
  # Extract text and save as .txt
  if file_ext == "docx":
    text = extract_docx(file_path)
  elif file_ext == "pdf":
    text = extract_pdf(file_path)
  else:
    text = extract_txt(file_path)
  text_filename = f"{os.path.splitext(save_name)[0]}.txt"
  text_path = os.path.join(applicant_dir, text_filename)
  with open(text_path, "w", encoding="utf-8") as f:
    f.write(text)
  return text_path

def build_tree(root="resume_and_supporting_docs"):
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

def resume_exists(applicant_dir, name_prefix):
  """
  Returns True if a resume file already exists in the applicant's folder.
  Looks for files like <name_prefix>_resume.<ext>
  """
  if not os.path.exists(applicant_dir):
    return False
  for fname in os.listdir(applicant_dir):
    if fname.startswith(f"{name_prefix}_resume."):
      return True
  return False

# --- UI FOR FILE TYPE SELECTION (CHANGED) ---
def get_file_type_selections(uploaded_files):
  """
  Let user specify which file is the resume (or none), and which are supporting docs.
  Returns: main_resume_idx (int or None), supporting_docs (list of bool)
  """
  st.write("### File Classification")
  file_labels = [f.name for f in uploaded_files]
  main_resume_idx = None

  if len(uploaded_files) > 1:
    options = ["None (all are supporting docs)"] + file_labels
    selection = st.radio(
      "Select which file is the main resume (if any):",
      options=list(range(len(options))),
      format_func=lambda i: options[i],
      index=0
    )
    if selection == 0:
      main_resume_idx = None
    else:
      main_resume_idx = selection - 1
  else:
    # Only one file: ask if it's a resume
    is_resume = st.checkbox(f"{file_labels[0]} is the main resume", value=True)
    main_resume_idx = 0 if is_resume else None

  # Supporting docs: all except main resume (or all if none selected)
  supporting_docs = []
  for i, f in enumerate(uploaded_files):
    if main_resume_idx is not None and i == main_resume_idx:
      supporting_docs.append(False)
    else:
      supporting_docs.append(True)
  return main_resume_idx, supporting_docs

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
  appl_name = ' '.join([part.title() for part in applicant_name.strip().split()])
  applicant_name = appl_name.strip()

  # --- CHANGES: File type selection UI ---
  if uploaded_files and applicant_name:
    names = applicant_name.strip().split()
    if len(names) < 1:
      st.warning("Please enter at least a first name.")
    else:
      name_prefix = "_".join(names)
      applicant_dir = os.path.join("resume_and_supporting_docs", name_prefix)
      # --- NEW: Check if resume already exists ---
      if resume_exists(applicant_dir, name_prefix):
        st.info("A resume already exists for this applicant. All uploaded files will be saved as supporting documents.")
        main_resume_idx = None
        supporting_docs = [True] * len(uploaded_files)
      else:
        main_resume_idx, supporting_docs = get_file_type_selections(uploaded_files)
  else:
    main_resume_idx, supporting_docs = None, []

  if st.button("Process Files"):
    if not uploaded_files:
      st.warning("There are no files to process :(")
    elif not applicant_name:
      st.warning("Applicant's name has not been entered :(")
    else:
      names = applicant_name.strip().split()
      if len(names) < 1:
        st.warning("Please enter at least a first name.")
      else:
        name_prefix = "_".join(names)
        applicant_dir = os.path.join("resume_and_supporting_docs", name_prefix)
        os.makedirs(applicant_dir, exist_ok=True)
        # Count existing supporting docs for numbering
        next_supp_num = get_next_supporting_number(applicant_dir, name_prefix)
        with st.spinner("Extracting and saving files..."):
          for idx, uploaded_file in enumerate(uploaded_files):
            file_ext = uploaded_file.name.split(".")[-1].lower()
            # Determine filename
            if main_resume_idx is not None and idx == main_resume_idx:
              save_name = f"{name_prefix}_resume.{file_ext}"
            elif supporting_docs and supporting_docs[idx]:
              save_name = f"{name_prefix}_supporting_{next_supp_num}.{file_ext}"
              next_supp_num += 1
            else:
              continue
            try:
              save_and_extract_file(uploaded_file, applicant_dir, save_name)
            except Exception as e:
              st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        st.success(f"Successfully processed {len(uploaded_files)} files for {applicant_name}!")

  # GitHub integration (unchanged)
  st.divider()
  st.subheader("GitHub Integration")
  if st.button("Push to GitHub"):
    with st.spinner("Pushing to GitHub..."):
      try:
        if not st.session_state.repo:
          st.session_state.repo = setup_git_repo()
        if st.session_state.repo:
          repo = st.session_state.repo
          repo.git.add("resume_and_supporting_docs/")
          repo.index.commit("Add new applicant files")
          origin = repo.remote(name="origin")
          origin.push()
          st.success("Files pushed to GitHub repository!")
      except Exception as e:
        st.error(f"Push failed: {str(e)}")

  # Display folder structure (unchanged)
  st.divider()
  st.subheader("📂 Uploaded Applicants's Resumes & Files")
  nodes = build_tree()
  if nodes:
    tree_select(nodes)
  else:
    st.info("No applicant files uploaded yet.")
