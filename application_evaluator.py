import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import time
import spacy
from spacy import displacy
import boto3
from botocore.exceptions import NoCredentialsError
import os
import requests
from datetime import datetime

# Set page config and background color
st.set_page_config(layout="wide", page_title="Onboarding Bot")
st.markdown("""
    <style>
    .stApp {
        background-color: #e6f7ff;  /* Baby blue background */
    }
    .progress-container {
        display: flex;
        height: 30px;
        margin-bottom: 20px;
        border-radius: 15px;
        overflow: hidden;
    }
    .progress-stage {
        flex: 1;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
    }
    .evaluation-section {
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'applicants' not in st.session_state:
    st.session_state.applicants = pd.DataFrame(columns=[
        'Name', 'Stage', 'Status', 'Resume', 'SOP', 'Coding', 'Interview', 
        'Resume_Score', 'Coding_Score', 'Interview_Score'
    ])
if 'current_applicant' not in st.session_state:
    st.session_state.current_applicant = None

# AWS Configuration (replace with your credentials)
AWS_ACCESS_KEY = st.secrets["AWS_ACCESS_KEY"]
AWS_SECRET_KEY = st.secrets["AWS_SECRET_SECRET"]
S3_BUCKET = "onboarding-bot-videos"

# Initialize NLP model
nlp = spacy.load("en_core_web_sm")

# Progress bar colors
STAGE_COLORS = {
    "Resume": "#FF69B4",  # Pink
    "Coding": "#FFFF00",   # Yellow
    "Interview": "#00FF00",# Green
    "Removed": "#FF0000"   # Red
}

# ========================
# Helper Functions
# ========================

def upload_to_s3(file, bucket_name, object_name):
    """Upload file to S3 bucket"""
    s3 = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY
    )
    try:
        s3.upload_fileobj(file, bucket_name, object_name)
        return f"https://{bucket_name}.s3.amazonaws.com/{object_name}"
    except (NoCredentialsError, Exception) as e:
        st.error(f"Error uploading file: {e}")
        return None

def transcribe_video(video_url):
    """Mock transcription function - replace with actual API call"""
    with st.spinner("Transcribing video..."):
        time.sleep(2)  # Simulate processing time
        return "This is a mock transcription. Replace with actual transcription API."

def evaluate_resume(text):
    """Evaluate resume using NLP"""
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    completeness = len(text) / 1000  # Simple metric
    return {
        "entities": entities,
        "completeness": min(completeness, 1.0),
        "score": min(len(text) / 500, 10)  # Max score 10
    }

def evaluate_coding(code):
    """Evaluate code quality"""
    # Simple heuristic-based evaluation
    complexity = code.count("for") + code.count("while") * 2
    comments_ratio = code.count("#") / max(1, len(code.split("\n")))
    return {
        "complexity": complexity,
        "comments_ratio": comments_ratio,
        "score": min(complexity + comments_ratio * 5, 10)
    }

def evaluate_interview(transcript):
    """Evaluate interview transcript"""
    doc = nlp(transcript)
    sentiment = doc.sentiment
    keywords = ["team", "experience", "skills", "passion"]
    keyword_count = sum(1 for word in doc if word.text.lower() in keywords)
    return {
        "sentiment": sentiment,
        "keyword_score": keyword_count,
        "score": min(keyword_count * 2, 10)
    }

# ========================
# UI Components
# ========================

def render_progress_bar(applicant):
    """Render colored progress bar based on applicant stage"""
    status = applicant["Status"] if not applicant.empty else "Resume"
    colors = []
    
    if status == "Removed":
        colors = [STAGE_COLORS["Removed"]] * 3
    else:
        for stage in ["Resume", "Coding", "Interview"]:
            if stage == status:
                colors.append(STAGE_COLORS[stage])
            elif list(STAGE_COLORS.keys()).index(stage) < list(STAGE_COLORS.keys()).index(status):
                colors.append(STAGE_COLORS[stage])
            else:
                colors.append("#CCCCCC")  # Gray for incomplete stages
    
    st.markdown(f"""
        <div class="progress-container">
            <div class="progress-stage" style="background-color:{colors[0]}">Resume & SOP</div>
            <div class="progress-stage" style="background-color:{colors[1]}">Coding Round</div>
            <div class="progress-stage" style="background-color:{colors[2]}">Interview</div>
        </div>
    """, unsafe_allow_html=True)

def applicant_selection():
    """Applicant selection dropdown and file upload"""
    st.header("Applicant Selection")
    
    # Upload CSV with applicant names
    uploaded_csv = st.file_uploader("Upload applicant list (CSV)", type="csv")
    if uploaded_csv:
        df = pd.read_csv(uploaded_csv)
        st.session_state.applicants = df
        
    # Applicant selection dropdown
    if not st.session_state.applicants.empty:
        applicant_names = st.session_state.applicants["Name"].tolist()
        selected_applicant = st.selectbox("Select Applicant", applicant_names)
        st.session_state.current_applicant = st.session_state.applicants[
            st.session_state.applicants["Name"] == selected_applicant
        ].iloc[0]
    else:
        st.warning("Upload a CSV file with applicant names")

def resume_stage():
    """Resume & SOP evaluation section"""
    st.header("Resume & SOP Evaluation")
    
    if st.session_state.current_applicant is None:
        st.warning("Select an applicant first")
        return
        
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload Documents")
        resume_file = st.file_uploader("Upload Resume (PDF/DOC)", type=["pdf", "docx"])
        sop_file = st.file_uploader("Upload Statement of Purpose", type=["pdf", "docx"])
        
        if st.button("Evaluate Resume & SOP"):
            if resume_file and sop_file:
                # Save to session state
                st.session_state.current_applicant["Resume"] = resume_file.name
                st.session_state.current_applicant["SOP"] = sop_file.name
                
                # Evaluate (mock implementation)
                evaluation = evaluate_resume(resume_file.read().decode("utf-8", errors="ignore"))
                
                # Update scores
                st.session_state.current_applicant["Resume_Score"] = evaluation["score"]
                
                st.success(f"Evaluation complete! Score: {evaluation['score']}/10")
            else:
                st.error("Upload both files first")
    
    with col2:
        st.subheader("Evaluation Results")
        if "Resume_Score" in st.session_state.current_applicant:
            score = st.session_state.current_applicant["Resume_Score"]
            st.metric("Overall Score", f"{score}/10")
            st.progress(score / 10)
            
            # Show extracted entities
            st.write("**Key Entities Found:**")
            st.json({
                "Skills": ["Python", "Machine Learning", "Data Analysis"],
                "Experience": ["5 years", "Senior Developer"],
                "Education": ["MIT Computer Science"]
            })
            
            # Decision buttons
            st.subheader("Decision")
            if st.button("❌ Remove Applicant", type="primary"):
                st.session_state.current_applicant["Status"] = "Removed"
                st.error("Applicant removed from process")
            if st.button("✅ Move to Coding Round", type="secondary"):
                st.session_state.current_applicant["Status"] = "Coding"
                st.success("Applicant moved to coding round")

def coding_stage():
    """Coding round evaluation section"""
    st.header("Coding Round Evaluation")
    
    if st.session_state.current_applicant is None:
        st.warning("Select an applicant first")
        return
        
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload Solution")
        code_file = st.file_uploader("Upload Code Solution", type=["py", "js", "java", "cpp"])
        
        if st.button("Evaluate Code"):
            if code_file:
                # Save to session state
                st.session_state.current_applicant["Coding"] = code_file.name
                
                # Evaluate (mock implementation)
                code_content = code_file.read().decode("utf-8")
                evaluation = evaluate_coding(code_content)
                
                # Update scores
                st.session_state.current_applicant["Coding_Score"] = evaluation["score"]
                
                st.success(f"Evaluation complete! Score: {evaluation['score']}/10")
            else:
                st.error("Upload code file first")
    
    with col2:
        st.subheader("Evaluation Results")
        if "Coding_Score" in st.session_state.current_applicant:
            score = st.session_state.current_applicant["Coding_Score"]
            st.metric("Code Quality Score", f"{score}/10")
            st.progress(score / 10)
            
            # Show evaluation metrics
            st.write("**Code Analysis:**")
            st.json({
                "Complexity": "High",
                "Readability": "Good",
                "Efficiency": "Optimal"
            })
            
            # Decision buttons
            st.subheader("Decision")
            if st.button("❌ Remove Applicant", key="coding_remove"):
                st.session_state.current_applicant["Status"] = "Removed"
                st.error("Applicant removed from process")
            if st.button("✅ Move to Interview", key="coding_move"):
                st.session_state.current_applicant["Status"] = "Interview"
                st.success("Applicant moved to interview round")

def interview_stage():
    """Interview round evaluation section"""
    st.header("Interview Round Evaluation")
    
    if st.session_state.current_applicant is None:
        st.warning("Select an applicant first")
        return
        
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload Interview")
        video_file = st.file_uploader("Upload Interview Video", type=["mp4", "mov"])
        
        if st.button("Process Interview"):
            if video_file:
                # Upload to S3
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                object_name = f"{st.session_state.current_applicant['Name']}_{timestamp}.mp4"
                video_url = upload_to_s3(video_file, S3_BUCKET, object_name)
                
                # Transcribe
                transcript = transcribe_video(video_url)
                
                # Evaluate
                evaluation = evaluate_interview(transcript)
                
                # Update scores
                st.session_state.current_applicant["Interview"] = video_url
                st.session_state.current_applicant["Interview_Score"] = evaluation["score"]
                
                st.success(f"Evaluation complete! Score: {evaluation['score']}/10")
                st.text_area("Transcript", transcript, height=300)
            else:
                st.error("Upload video file first")
    
    with col2:
        st.subheader("Evaluation Results")
        if "Interview_Score" in st.session_state.current_applicant:
            score = st.session_state.current_applicant["Interview_Score"]
            st.metric("Interview Score", f"{score}/10")
            st.progress(score / 10)
            
            # Show sentiment analysis
            st.write("**Sentiment Analysis:**")
            st.json({
                "Positive": 0.75,
                "Neutral": 0.20,
                "Negative": 0.05
            })
            
            # Decision buttons
            st.subheader("Final Decision")
            if st.button("❌ Reject Applicant", key="interview_reject"):
                st.session_state.current_applicant["Status"] = "Removed"
                st.error("Applicant rejected")
            if st.button("✅ Hire Applicant", key="interview_hire"):
                st.session_state.current_applicant["Status"] = "Hired"
                st.balloons()
                st.success("Applicant hired!")

# ========================
# Main App
# ========================

def main():
    st.title("🚀 Automated Onboarding Bot")
    
    # Applicant selection
    applicant_selection()
    
    # Progress bar
    if st.session_state.current_applicant is not None:
        render_progress_bar(st.session_state.current_applicant)
        
        # Show current stage
        current_stage = st.session_state.current_applicant["Status"]
        st.subheader(f"Current Stage: {current_stage}")
        
        # Stage routing
        if current_stage == "Resume" or current_stage == "New":
            resume_stage()
        elif current_stage == "Coding":
            coding_stage()
        elif current_stage == "Interview":
            interview_stage()
        elif current_stage == "Removed":
            st.error("This applicant has been removed from the process")
        elif current_stage == "Hired":
            st.balloons()
            st.success("This applicant has been hired!")
    else:
        st.info("Select an applicant to begin evaluation")

if __name__ == "__main__":
    main()
