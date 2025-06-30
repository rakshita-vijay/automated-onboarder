import streamlit as st
import pandas as pd
from io import BytesIO
import base64
import time
import spacy
from spacy import displacy
import requests
from bs4 import BeautifulSoup
import os
import tempfile

# Initialize NLP model
nlp = spacy.load("en_core_web_sm")

# Page configuration
st.set_page_config(
    page_title="Onboarding Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for background and progress bars
st.markdown(
    """
    <style>
    body {
        background-color: #e6f7ff; /* Baby blue background */
    }
    .stProgress > div > div > div {
        background-color: #ff69b4; /* Pink for stage 1 */
    }
    .stage-2 > div > div > div {
        background-color: #ffff00; /* Yellow for stage 2 */
    }
    .stage-3 > div > div > div {
        background-color: #00ff00; /* Green for stage 3 */
    }
    .rejected > div > div > div {
        background-color: #ff0000; /* Red for rejected */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Session state initialization
if 'applicants' not in st.session_state:
    st.session_state.applicants = pd.DataFrame(columns=['Name', 'Stage', 'Status', 'Resume', 'Code', 'Interview'])
if 'selected_applicant' not in st.session_state:
    st.session_state.selected_applicant = None
if 'current_stage' not in st.session_state:
    st.session_state.current_stage = 1

# Applicant data management
def load_applicants(uploaded_file):
    if uploaded_file:
        st.session_state.applicants = pd.read_csv(uploaded_file)

# Progress bar display
def show_progress_bar():
    if st.session_state.selected_applicant:
        applicant = st.session_state.applicants[
            st.session_state.applicants['Name'] == st.session_state.selected_applicant
        ].iloc[0]
        
        status = applicant['Status']
        current_stage = applicant['Stage']
        
        # Determine progress bar color
        bar_class = "rejected" if status == "Rejected" else f"stage-{current_stage}"
        
        # Create progress bar
        progress_html = f"""
        <div class="stProgress {bar_class}" style="height: 30px; border-radius: 5px; margin-bottom: 20px;">
            <div style="width: {current_stage * 33}%; height: 100%;"></div>
        </div>
        """
        st.markdown(progress_html, unsafe_allow_html=True)

# File processing functions
def extract_text_from_pdf(uploaded_file):
    # Placeholder for actual PDF extraction
    return "PDF content would be extracted here..."

def transcribe_video(uploaded_file):
    # Placeholder for actual transcription
    return "Video transcription would appear here..."

# Evaluation functions
def evaluate_resume(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    
    # Fact-checking simulation
    facts = []
    for ent in doc.ents:
        if ent.label_ in ["ORG", "PERSON"]:
            search_url = f"https://www.google.com/search?q={ent.text}"
            facts.append(f"Fact-checked: {ent.text} (Source: {search_url})")
    
    return {
        "entities": entities,
        "completeness": len(text) / 1000,
        "meaningfulness": len(doc.sents) / 10,
        "facts": facts
    }

def evaluate_code(uploaded_file):
    # Placeholder for actual code evaluation
    return {
        "working": True,
        "completeness": 1.0,
        "score": 8.5
    }

def evaluate_interview(transcript):
    doc = nlp(transcript)
    sentiment = 0
    for sentence in doc.sents:
        # Simple sentiment analysis
        if "excellent" in sentence.text.lower() or "good" in sentence.text.lower():
            sentiment += 1
    
    return {
        "sentiment": sentiment / len(list(doc.sents)) * 10,
        "completeness": len(transcript) / 500,
        "key_points": [chunk.text for chunk in doc.noun_chunks]
    }

# Main app
st.title("🤖 Automated Onboarding Bot")

# Applicant selection
st.sidebar.header("Applicant Management")
uploaded_file = st.sidebar.file_uploader("Upload Applicant CSV", type="csv")
if uploaded_file:
    load_applicants(uploaded_file)

if not st.session_state.applicants.empty:
    applicant_names = st.session_state.applicants['Name'].tolist()
    st.session_state.selected_applicant = st.sidebar.selectbox(
        "Select Applicant",
        applicant_names
    )
    
    # Display progress bar
    show_progress_bar()
    
    # Get current applicant data
    current_applicant = st.session_state.applicants[
        st.session_state.applicants['Name'] == st.session_state.selected_applicant
    ].iloc[0]
    
    # Stage 1: Resume & SOP
    st.header("📄 Stage 1: Resume & SOP Evaluation")
    resume_file = st.file_uploader("Upload Resume (PDF/DOC)", type=["pdf", "docx"])
    
    if resume_file:
        resume_text = extract_text_from_pdf(resume_file)
        evaluation = evaluate_resume(resume_text)
        
        st.subheader("Resume Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Completeness", f"{evaluation['completeness']:.2f}/1.0")
            st.metric("Meaningfulness", f"{evaluation['meaningfulness']:.2f}/10")
        with col2:
            st.write("**Key Entities:**")
            for entity, label in evaluation['entities']:
                st.write(f"- {entity} ({label})")
        
        st.subheader("Fact-Checking Results")
        for fact in evaluation['facts']:
            st.write(f"- {fact}")
    
    # Stage 2: Coding Round
    st.header("💻 Stage 2: Coding Assignment")
    code_files = st.file_uploader("Upload Code Files", accept_multiple_files=True, type=["py", "java", "js", "cpp"])
    
    if code_files:
        evaluation = evaluate_code(code_files)
        
        st.subheader("Code Evaluation")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Working Solution", "Yes" if evaluation['working'] else "No")
            st.metric("Completeness", f"{evaluation['completeness']:.2f}/1.0")
        with col2:
            st.metric("Score", f"{evaluation['score']}/10")
    
    # Stage 3: Interview Round
    st.header("🎥 Stage 3: Interview Evaluation")
    video_file = st.file_uploader("Upload Interview Video", type=["mp4", "mov"])
    
    if video_file:
        transcript = transcribe_video(video_file)
        evaluation = evaluate_interview(transcript)
        
        st.subheader("Transcript Analysis")
        st.text_area("Transcript", transcript, height=200)
        
        st.subheader("Evaluation Results")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Sentiment Score", f"{evaluation['sentiment']:.2f}/10")
            st.metric("Completeness", f"{evaluation['completeness']:.2f}/1.0")
        with col2:
            st.write("**Key Points Discussed:**")
            for point in evaluation['key_points'][:5]:
                st.write(f"- {point}")
    
    # Decision buttons
    st.header("Decision")
    col1, col2, col3 = st.columns([1,1,2])
    with col1:
        if st.button("🚫 Remove Applicant"):
            st.session_state.applicants.loc[
                st.session_state.applicants['Name'] == st.session_state.selected_applicant,
                'Status'
            ] = "Rejected"
            st.experimental_rerun()
    
    with col2:
        if st.button("✅ Move to Next Stage"):
            current_stage = st.session_state.applicants.loc[
                st.session_state.applicants['Name'] == st.session_state.selected_applicant,
                'Stage'
            ].values[0]
            if current_stage < 3:
                st.session_state.applicants.loc[
                    st.session_state.applicants['Name'] == st.session_state.selected_applicant,
                    'Stage'
                ] = current_stage + 1
                st.experimental_rerun()
    
    # Save updated data
    st.sidebar.download_button(
        "Download Updated Applicant Data",
        st.session_state.applicants.to_csv(index=False),
        "applicant_data.csv",
        "text/csv"
    )
else:
    st.info("Upload a CSV file containing applicant data to get started")

# Memory entries integration
st.sidebar.header("System Memory")
st.sidebar.write("UI Development: Building Streamlit components with progress bars and spinners")
st.sidebar.write("AI Interface: Customizing progress bars and color schemes")
st.sidebar.write("Bot Development: Creating AI chatbots with Python integration")
st.sidebar.write("Algorithms: Processing applicant data with NLP and evaluation algorithms")
