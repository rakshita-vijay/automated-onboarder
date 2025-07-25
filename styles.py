css_dark = """
<style>
/* ===== BODY & GENERAL LAYOUT ===== */ 
@media (prefers-color-scheme: dark) {
    body { background: #181920 !important; font-family: 'Segoe UI', sans-serif; color: #fff !important; }
}

/* ===== MAIN HEADING ===== */ 
@media (prefers-color-scheme: dark) {
    h1, .stTitle { color: #fff !important; font-weight: 900; letter-spacing: 1px; white-space: nowrap; text-align: center; }
}

/* ===== TEXT INPUT FIELDS ===== */ 
@media (prefers-color-scheme: dark) {
    [data-testid="stTextInput"] input {
        background-color: #181920 !important;
        color: #fff !important;
        border: 1.5px solid #fff !important;
        border-radius: 10px !important;
        font-size: 1.1rem !important;
    }
    [data-testid="stTextInput"] input::placeholder { color: #aaa !important; opacity: 1 !important; font-style: italic; }
    [data-testid="stTextInput"] label { color: #fff !important; font-weight: bold; }
}

/* ===== FILE UPLOADER CONTAINER ===== */ 
@media (prefers-color-scheme: dark) {
    .stFileUploader {
        border: 2px dashed #fff !important;
        background: #181920 !important;
        border-radius: 12px !important;
    }
    [data-testid="stFileUploader"] > div > div {
        background-color: #181920 !important;
        border-radius: 10px !important;
    }
}

/* ===== FILE UPLOADER DROPZONE TEXT ===== */ 
@media (prefers-color-scheme: dark) {
    [data-testid="stFileUploaderDropzone"] * { color: #fff !important; opacity: 1 !important; text-shadow: 0 0 2px #000; }
    [data-testid="stFileUploader"] button {
        color: #fff !important; border-color: #fff !important; background-color: #181920 !important;
    }
}

/* ===== UPLOADED FILE INFO DISPLAY ===== */ 
@media (prefers-color-scheme: dark) {
    [data-testid="stFileUploader"] .uploadedFile,
    [data-testid="stFileUploader"] [data-testid="fileStatus"],
    [data-testid="stFileUploader"] div[data-testid="stFileUploaderDropzone"] + div {
        background-color: #181920 !important;
        color: #fff !important;
        border: 1px solid #fff !important;
        border-radius: 8px !important;
        padding: 8px !important;
        margin: 5px 0 !important;
    }
}

/* ===== PROCESS FILES BUTTON ===== */ 
@media (prefers-color-scheme: dark) {
    .stButton>button {
        background: #181920 !important;
        color: #fff !important;
        border: 2px solid #fff !important;
        font-weight: bold;
        border-radius: 6px;
        font-size: 1.1rem;
        padding: 0.5em 2em;
        margin-top: 0.5em;
        transition: background 0.3s;
    }
    .stButton>button:hover {
        background: #333 !important;
        color: #fff !important;
        border: 2px solid #fff !important;
    }
}

/* ===== ALERTS & NOTIFICATIONS ===== */ 
@media (prefers-color-scheme: dark) {
    .stAlert {
        background-color: #181920 !important;
        color: #fff !important;
        border: 1px solid #fff !important;
        border-radius: 8px;
    }
}

/* ===== FOLDER STRUCTURE DISPLAY ===== */ 
@media (prefers-color-scheme: dark) {
    .folder-box {
        background: #181920 !important;
        color: #fff !important;
        border: 1px solid #fff !important;
        border-radius: 10px;
        box-shadow: 0 2px 8px #000;
        padding: 1em;
        margin-bottom: 1em;
    }
    .stMarkdown ul {
        margin-left: 1.5em;
        color: #fff !important;
    }
}

/* ===== STREAMLIT SIDEBAR (if used) ===== */ 
@media (prefers-color-scheme: dark) {
    .css-1d391kg {
        background-color: #181920 !important;
        color: #fff !important;
    }
}

/* Make st.page_link look like outlined buttons */
.stPageLink {
    display: block; 
    max-width: 300px; 
    margin: 8px 0;
    border: 2px solid #fff;
    border-radius: 10px;
    background: transparent;
    padding: 8px 16px;
    color: #fff;
    font-size: 1.2em;
    font-weight: bold;
    text-decoration: none;
    transition: background 0.2s, color 0.2s, box-shadow 0.2s;
    box-shadow: 0 2px 8px rgba(108,99,255,0.05);
    cursor: pointer;
}
.stPageLink:hover {
    background: #333 !important;
    color: #fff !important;
    border: 2px solid #fff !important; 
}
.stPageLink a {
    text-decoration: none !important;
    color: inherit !important;
    display: block;
}
/* Make all text in the tree select widget white in dark mode */
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




# css_light 






/* ===== BODY & GENERAL LAYOUT ===== */
body[data-theme="light"] b { background: #ffffff !important; font-family: 'Segoe UI', sans-serif; color: #000 !important; }

/* ===== MAIN HEADING ===== */ 
body[data-theme="light"] .stHeading,
body[data-theme="light"] h1,
body[data-theme="light"] .stTitle {
    color: #000 !important;
    background: #fff !important;
    font-weight: 900;
    letter-spacing: 1px;
    white-space: nowrap;
    text-align: center;
}

/* ===== TEXT INPUT FIELDS ===== */
body[data-theme="light"] [data-testid="stTextInput"] input {
    background-color: #e6f7ff !important;
    border-radius: 10px !important;
    color: #222 !important;
    border: 1.5px solid #a5b4fc !important;
    font-size: 1.1rem !important;
}

body[data-theme="light"] [data-testid="stTextInput"] input::placeholder {
    color: #666 !important;
    opacity: 1 !important;
    font-style: italic;
}

body[data-theme="light"] [data-testid="stTextInput"] label {
    color: #000 !important;
    font-weight: bold;
}

/* ===== FILE UPLOADER CONTAINER ===== */
body[data-theme="light"] .stFileUploader {
    border: 2px dashed #000 !important;
    background: #ffffff !important;
    border-radius: 12px !important;
}

body[data-theme="light"] [data-testid="stFileUploader"] > div > div {
    background-color: #ffffff !important;
    border-radius: 10px !important;
}

/* ===== FILE UPLOADER DROPZONE TEXT ===== */
body[data-theme="light"] [data-testid="stFileUploaderDropzone"] * {
    color: #000 !important;
    opacity: 1 !important;
    text-shadow: none !important;
}

body[data-theme="light"] [data-testid="stFileUploader"] button {
    color: #000 !important;
    border-color: #000 !important;
    background-color: #fff !important;
}

/* ===== UPLOADED FILE INFO DISPLAY ===== */ 
body[data-theme="light"] [data-testid="stFileUploader"] .uploadedFile,
body[data-theme="light"] [data-testid="stFileUploader"] [data-testid="fileStatus"],
body[data-theme="light"] [data-testid="stFileUploader"] div[data-testid="stFileUploaderDropzone"] + div {
    background-color: #ffffff !important;
    color: #000 !important;
    border: 1px solid #000 !important;
    border-radius: 8px !important;
    padding: 8px !important;
    margin: 5px 0 !important;
}

/* ===== PROCESS FILES BUTTON ===== */ 
body[data-theme="light"] .stButton > button {
    background: #ffffff !important;
    color: #000 !important;
    border: 2px solid #000 !important;
    font-weight: bold;
    border-radius: 6px;
    font-size: 1.1rem;
    padding: 0.5em 2em;
    margin-top: 0.5em;
    transition: background 0.3s;
}

body[data-theme="light"] .stButton > button:hover {
    background: #f0f0f0 !important;
    color: #000 !important;
    border: 2px solid #000 !important;
}

/* ===== ALERTS & NOTIFICATIONS ===== */ 
body[data-theme="light"] .stAlert {
    background-color: #ffffff !important;
    color: #000 !important;
    border: 1px solid #000 !important;
    border-radius: 8px;
}

/* ===== FOLDER STRUCTURE DISPLAY ===== */ 
body[data-theme="light"] .folder-box {
    background: #ffffff !important;
    color: #000 !important;
    border: 1px solid #000 !important;
    border-radius: 10px;
    box-shadow: 0 2px 8px #ddd;
    padding: 1em;
    margin-bottom: 1em;
}

body[data-theme="light"] .stMarkdown ul {
    margin-left: 1.5em;
    color: #000 !important;
}

/* ===== STREAMLIT SIDEBAR (if used) ===== */
body[data-theme="light"] .css-1d391kg {
    background-color: #ffffff !important;
    color: #000 !important;
}

/* Make all text in the tree select widget black in light mode */
body[data-theme="light"] [data-testid="stTreeSelect"] span,
body[data-theme="light"] [data-testid="stTreeSelect"] label,
body[data-theme="light"] [data-testid="stTreeSelect"] div {
    color: #000 !important;
}

body[data-theme="light"] [data-testid="stTreeSelect"] input[type="checkbox"] {
    filter: invert(1); /* Optional: makes the checkbox white */
}
</style> 
"""
















css_perfect_black = """  
<style>
/* ===== BODY & GENERAL LAYOUT ===== */
/* Light Mode: White background */
@media (prefers-color-scheme: light) {
    body {
        background: #ffffff !important;
        font-family: 'Segoe UI', sans-serif;
        color: #000 !important;
    }
}
/* Dark Mode: Dark background */
@media (prefers-color-scheme: dark) {
    body {
        background: #181920 !important; /* ADDED: Dark background */
        font-family: 'Segoe UI', sans-serif;
        color: #fff !important; /* ADDED: White text */
    }
}

/* ===== MAIN HEADING ===== */
/* Light Mode: Dark heading */
@media (prefers-color-scheme: light) {
    h1, .stTitle {
        color: #000 !important;
        font-weight: 900;
        letter-spacing: 1px;
        white-space: nowrap;
        text-align: center;
    }
}
/* Dark Mode: White heading */
@media (prefers-color-scheme: dark) {
    h1, .stTitle {
        color: #fff !important; /* ADDED: White heading in dark mode */
        font-weight: 900;
        letter-spacing: 1px;
        white-space: nowrap;
        text-align: center;
    }
}

/* ===== TEXT INPUT FIELDS (Applicant Name) ===== */
/* Light Mode: White background, black text, black border */
@media (prefers-color-scheme: light) {
    [data-testid="stTextInput"] input {
        background-color: #ffffff !important;
        color: #000 !important;
        border: 1.5px solid #000 !important;
        border-radius: 10px !important;
        font-size: 1.1rem !important;
    }
    [data-testid="stTextInput"] input::placeholder {
        color: #666 !important;
        opacity: 1 !important;
        font-style: italic;
    }
    [data-testid="stTextInput"] label {
        color: #000 !important; /* ADDED: Black label */
        font-weight: bold;
    }
}
/* Dark Mode: Dark background, white text, white border */
@media (prefers-color-scheme: dark) {
    [data-testid="stTextInput"] input {
        background-color: #181920 !important; /* ADDED: Dark background */
        color: #fff !important; /* ADDED: White text */
        border: 1.5px solid #fff !important; /* ADDED: White border */
        border-radius: 10px !important;
        font-size: 1.1rem !important;
    }
    [data-testid="stTextInput"] input::placeholder {
        color: #aaa !important; /* ADDED: Light gray placeholder */
        opacity: 1 !important;
        font-style: italic;
    }
    [data-testid="stTextInput"] label {
        color: #fff !important; /* ADDED: White label */
        font-weight: bold;
    }
}

/* ===== FILE UPLOADER CONTAINER ===== */
/* Light Mode: White background, black dashed border */
@media (prefers-color-scheme: light) {
    .stFileUploader {
        border: 2px dashed #000 !important; /* ADDED: Black dashed border */
        background: #ffffff !important;
        border-radius: 12px !important;
    }
    [data-testid="stFileUploader"] > div > div {
        background-color: #ffffff !important;
        border-radius: 10px !important;
    }
}
/* Dark Mode: Dark background, white dashed border */
@media (prefers-color-scheme: dark) {
    .stFileUploader {
        border: 2px dashed #fff !important; /* ADDED: White dashed border */
        background: #181920 !important; /* ADDED: Dark background */
        border-radius: 12px !important;
    }
    [data-testid="stFileUploader"] > div > div {
        background-color: #181920 !important; /* ADDED: Dark background */
        border-radius: 10px !important;
    }
}

/* ===== FILE UPLOADER DROPZONE TEXT ===== */
/* Light Mode: Black text */Add commentMore actions
@media (prefers-color-scheme: light) {
    [data-testid="stFileUploaderDropzone"] * {
        color: #000 !important;
        opacity: 1 !important;
        text-shadow: none !important;
    }
    [data-testid="stFileUploader"] button {
        color: #000 !important; /* ADDED: Black button text */
        border-color: #000 !important; /* ADDED: Black button border */
        background-color: #fff !important; /* ADDED: White button background */
    }
}
/* Dark Mode: White text */
@media (prefers-color-scheme: dark) {
    [data-testid="stFileUploaderDropzone"] * {
        color: #fff !important; /* ADDED: White dropzone text */
        opacity: 1 !important;
        text-shadow: 0 0 2px #000; /* ADDED: Subtle glow for readability */
    }
    [data-testid="stFileUploader"] button {
        color: #fff !important; /* ADDED: White button text */
        border-color: #fff !important; /* ADDED: White button border */
        background-color: #181920 !important; /* ADDED: Dark button background */
    }
}

/* ===== UPLOADED FILE INFO DISPLAY ===== */
/* Light Mode: White background, black text */
@media (prefers-color-scheme: light) {
    [data-testid="stFileUploader"] .uploadedFile,
    [data-testid="stFileUploader"] [data-testid="fileStatus"],
    [data-testid="stFileUploader"] div[data-testid="stFileUploaderDropzone"] + div {
        background-color: #ffffff !important;
        color: #000 !important;
        border: 1px solid #000 !important; /* ADDED: Black border */
        border-radius: 8px !important; 
        padding: 8px !important;
        margin: 5px 0 !important;
    }
}
/* Dark Mode: Dark background, white text */
@media (prefers-color-scheme: dark) {
    [data-testid="stFileUploader"] .uploadedFile,
    [data-testid="stFileUploader"] [data-testid="fileStatus"],
    [data-testid="stFileUploader"] div[data-testid="stFileUploaderDropzone"] + div {
        background-color: #181920 !important; /* ADDED: Dark background */
        color: #fff !important; /* ADDED: White text */
        border: 1px solid #fff !important; /* ADDED: White border */
        border-radius: 8px !important;
        padding: 8px !important;
        margin: 5px 0 !important;
    }
}

/* ===== PROCESS FILES BUTTON ===== */
/* Light Mode: White background, black text */
@media (prefers-color-scheme: light) {
    .stButton>button {
        background: #ffffff !important; /* ADDED: White background */
        color: #000 !important; /* ADDED: Black text */
        border: 2px solid #000 !important; /* ADDED: Black border */
        font-weight: bold;
        border-radius: 6px;
        font-size: 1.1rem;
        padding: 0.5em 2em;
        margin-top: 0.5em;
        transition: background 0.3s;
    }
    .stButton>button:hover {
        background: #f0f0f0 !important; /* ADDED: Light gray hover */
        color: #000 !important;
        border: 2px solid #000 !important;
    }
}
/* Dark Mode: Dark background, white text */
@media (prefers-color-scheme: dark) {
    .stButton>button {
        background: #181920 !important; /* ADDED: Dark background */
        color: #fff !important; /* ADDED: White text */
        border: 2px solid #fff !important; /* ADDED: White border */
        font-weight: bold;
        border-radius: 6px;
        font-size: 1.1rem;
        padding: 0.5em 2em;
        margin-top: 0.5em;
        transition: background 0.3s;
    }
    .stButton>button:hover {
        background: #333 !important; /* ADDED: Darker hover */
        color: #fff !important;
        border: 2px solid #fff !important;
    }
}

/* ===== ALERTS & NOTIFICATIONS ===== */
/* Light Mode: White background, black text */
@media (prefers-color-scheme: light) {
    .stAlert {
        background-color: #ffffff !important; /* ADDED: White background */
        color: #000 !important; /* ADDED: Black text */
        border: 1px solid #000 !important; /* ADDED: Black border */
        border-radius: 8px;
    }
}
/* Dark Mode: Dark background, white text */
@media (prefers-color-scheme: dark) {
    .stAlert {
        background-color: #181920 !important; /* ADDED: Dark background */
        color: #fff !important; /* ADDED: White text */
        border: 1px solid #fff !important; /* ADDED: White border */
        border-radius: 8px;
    }
}

/* ===== FOLDER STRUCTURE DISPLAY ===== */
/* Light Mode: White background, black text */
@media (prefers-color-scheme: light) {
    .folder-box {
        background: #ffffff !important;
        color: #000 !important; /* ADDED: Black text */
        border: 1px solid #000 !important; /* ADDED: Black border */
        border-radius: 10px;
        box-shadow: 0 2px 8px #ddd; /* ADDED: Light shadow */
        padding: 1em;
        margin-bottom: 1em;
    }
    .stMarkdown ul {
        margin-left: 1.5em;
        color: #000 !important; /* ADDED: Black text for lists */
    }
}
/* Dark Mode: Dark background, white text */
@media (prefers-color-scheme: dark) {
    .folder-box {
        background: #181920 !important; /* ADDED: Dark background */
        color: #fff !important; /* ADDED: White text */
        border: 1px solid #fff !important; /* ADDED: White border */
        border-radius: 10px;
        box-shadow: 0 2px 8px #000; /* ADDED: Dark shadow */
        padding: 1em;
        margin-bottom: 1em;
    }
    .stMarkdown ul {
        margin-left: 1.5em;
        color: #fff !important; /* ADDED: White text for lists */
    }
}

/* ===== STREAMLIT SIDEBAR (if used) ===== */
/* Light Mode: White sidebar */
@media (prefers-color-scheme: light) {
    .css-1d391kg { /* ADDED: Streamlit sidebar selector */
        background-color: #ffffff !important;
        color: #000 !important;
    }
}
/* Dark Mode: Dark sidebar */
@media (prefers-color-scheme: dark) {
    .css-1d391kg { /* ADDED: Streamlit sidebar selector */
        background-color: #181920 !important;
        color: #fff !important;
    }
}
</style>
"""


'''
css_perfect_white = """
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
 
</style>
""" 
'''
