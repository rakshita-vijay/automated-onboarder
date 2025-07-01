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
</style>
"""
