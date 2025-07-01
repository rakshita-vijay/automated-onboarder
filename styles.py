# styles.py
css = """
<style>
body {
    background: linear-gradient(135deg, #e0e7ff 0%, #f0f9ff 100%);
    font-family: 'Segoe UI', sans-serif;
}
h1, .stTitle {
    color: #6c63ff;
    font-weight: 900;
    letter-spacing: 1px;
    white-space: nowrap;
    text-align: center;
}
.stTextInput>div>div>input {
    background-color: #f5f7fa;
    border-radius: 8px;
    border: 1.5px solid #a5b4fc;
    font-size: 1.1rem;
}
.stButton>button {
    background: linear-gradient(90deg, #6c63ff 0%, #48bb78 100%);
    color: white;
    font-weight: bold;
    border-radius: 6px;
    border: none;
    font-size: 1.1rem;
    padding: 0.5em 2em;
    margin-top: 0.5em;
    transition: background 0.3s;
}
.stButton>button:hover {
    background: linear-gradient(90deg, #48bb78 0%, #6c63ff 100%);
    color: #fff;
}
.stFileUploader {
    border: 2px dashed #6c63ff !important;
    background: #f0f9ff !important;
    border-radius: 12px !important;
}
.stAlert {
    border-radius: 8px;
}
.stMarkdown ul {
    margin-left: 1.5em;
}
.folder-box {
    background: #fff;
    border-radius: 10px;
    box-shadow: 0 2px 8px #dbeafe;
    padding: 1em;
    margin-bottom: 1em;
}  
/* File uploader area */
[data-testid="stFileUploader"] > div > div {
    background-color: #e6f7ff !important;
    border-radius: 10px !important;
}

/* Text input fields (Applicant Name, etc.) */
[data-testid="stTextInput"] input {
    background-color: #e6f7ff !important;
    border-radius: 10px !important;
    color: #222 !important;
    border: 1.5px solid #a5b4fc !important;
    font-size: 1.1rem !important;
} 
/* Fix applicant name placeholder text visibility in dark mode */
[data-testid="stTextInput"] input::placeholder {
    color: #666 !important;
    opacity: 1 !important;
    font-style: italic;
}

/* Fix uploaded file name and size visibility in dark mode */
[data-testid="stFileUploader"] .uploadedFile {
    background-color: #f0f9ff !important;
    color: #222 !important;
    border-radius: 8px !important;
    padding: 8px !important;
    margin: 5px 0 !important;
}

/* Target the file info text specifically */
[data-testid="stFileUploader"] [data-testid="fileStatus"] {
    color: #222 !important;
    background-color: #f0f9ff !important;
    padding: 8px !important;
    border-radius: 8px !important;
}

/* Alternative selector for uploaded file display */
[data-testid="stFileUploader"] div[data-testid="stFileUploaderDropzone"] + div {
    background-color: #f0f9ff !important;
    color: #222 !important;
    border-radius: 8px !important;
    padding: 8px !important;
} 

/* Make sure all text in file uploader area is dark */
[data-testid="stFileUploader"] * {
    color: #222 !important;
}

/* Specific targeting for file name and size text */
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] span {
    color: #222 !important;
    background-color: transparent !important;
}
 
/* --- Dropzone text color: black in light mode, white in dark mode --- */
@media (prefers-color-scheme: dark) {
  [data-testid="stFileUploaderDropzone"] * {
    color: #fff !important;
  }
  [data-testid="stFileUploader"] button {
    color: #fff !important;
    border-color: #fff !important;
  }
}
@media (prefers-color-scheme: light) {
  [data-testid="stFileUploaderDropzone"] * {
    color: #111 !important;
  }
  [data-testid="stFileUploader"] button {
    color: #111 !important;
    border-color: #111 !important;
  }
} 
@media (prefers-color-scheme: dark) {
  [data-testid="stFileUploaderDropzone"] * {
    color: #fff !important;
    opacity: 1 !important;         /* <-- Force fully visible */
    text-shadow: 0 0 2px #111;     /* Optional: add a subtle glow for readability */
  }
@media (prefers-color-scheme: dark) {
  [data-testid="stFileUploaderDropzone"] * {
    color: #fff !important;      /* Make text white */
    opacity: 1 !important;       /* Make text fully opaque */
    text-shadow: 0 0 2px #000;   /* Optional: subtle glow for readability */
  }
}
@media (prefers-color-scheme: light) {
  [data-testid="stFileUploaderDropzone"] * {
    color: #111 !important;      /* Make text black in light mode */
    opacity: 1 !important;
    text-shadow: none !important;
  }
}


</style>
""" 
