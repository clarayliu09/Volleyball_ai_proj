import streamlit as st
import os
from utils.video_processor import VideoProcessor
from utils.pose_analyzer import PoseAnalyzer
from utils.gpt_analyzer import GPTAnalyzer
import cv2
from PIL import Image
import numpy as np
import tempfile
import shutil
import openai
from dotenv import load_dotenv
import pandas as pd

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(page_title="Volleyball Serving Analysis",
                   page_icon="🏐",
                   layout="wide")

# Initialize session state
if 'metrics' not in st.session_state:
    st.session_state.metrics = []
if 'current_frame' not in st.session_state:
    st.session_state.current_frame = None
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'annotated_video_path' not in st.session_state:
    st.session_state.annotated_video_path = None
if 'temp_video_path' not in st.session_state:
    st.session_state.temp_video_path = None
if 'analysis_prompt' not in st.session_state:
    st.session_state.analysis_prompt = None

# Initialize components
video_processor = VideoProcessor()
pose_analyzer = PoseAnalyzer()
gpt_analyzer = GPTAnalyzer()

# Sidebar for OpenAI API key
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    if api_key:
        openai.api_key = api_key
        st.success("API key configured!")
    else:
        st.warning("Please enter your OpenAI API key to get analysis.")

# Title and description
st.title("🏐 Volleyball Serving Analysis Assistant")
st.markdown("""
This application analyzes volleyball serving techniques using computer vision and AI.
Upload a video or provide a YouTube URL to get started!
""")

# Input method selection
input_method = st.radio("Choose input method:",
                        ("YouTube URL", "Upload Video"))

# Process video function
def process_video(video_path):
    st.session_state.metrics = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # List to store all annotated frames
    annotated_frames = []

    # Get video properties
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Process frames
    for frame_idx, frame in video_processor.extract_video_frames(
            video_path, frame_interval=3):
        progress = frame_idx / total_frames
        progress_bar.progress(progress)
        status_text.text(f"Processing frame {frame_idx}/{total_frames}")

        # Analyze frame
        annotated_frame, metrics = pose_analyzer.analyze_frame(frame)
        if metrics["pose_detected"]:
            st.session_state.metrics.append(metrics)
        st.session_state.current_frame = annotated_frame
        annotated_frames.append(annotated_frame)

    progress_bar.progress(1.0)
    status_text.text("Processing complete!")
    
    # Save the annotated video
    if annotated_frames:
        st.session_state.annotated_video_path = video_processor.save_annotated_video(
            video_path, annotated_frames)

    # Generate analysis prompt and result
    if st.session_state.metrics:
        st.session_state.analysis_prompt = pose_analyzer.generate_analysis_prompt(
            st.session_state.metrics)
        st.write("Debug: Analysis prompt generated")
        
        if openai.api_key:
            try:
                st.session_state.analysis_result = gpt_analyzer.analyze_technique(
                    st.session_state.analysis_prompt)
                st.write("Debug: Analysis result generated")
            except Exception as e:
                st.error(f"Error generating AI analysis: {str(e)}")
                st.session_state.analysis_result = None
        else:
            st.warning("OpenAI API key not found. Skipping AI analysis.")

# Handle input
if input_method == "YouTube URL":
    url = st.text_input("Enter YouTube URL:")
    if url:
        if st.button("Analyze"):
            with st.spinner("Downloading video..."):
                video_path, error_msg = video_processor.download_video(url)
                if video_path:
                    process_video(video_path)
                else:
                    st.error(f"Error downloading video: {error_msg}")
                    st.info("""Common issues and solutions:
1. Make sure the video URL is correct and accessible
2. Check if the video is available in your region
3. Try a different video if the issue persists""")

else:  # Upload Video
    uploaded_file = st.file_uploader("Upload a video file",
                                     type=["mp4", "mov", "avi"])
    if uploaded_file:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False,
                                         suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name
            st.session_state.temp_video_path = video_path

        if st.button("Analyze"):
            process_video(video_path)
            # Clean up temporary file
            try:
                os.unlink(video_path)
            except:
                pass  # Ignore cleanup errors

# Display results
if st.session_state.annotated_video_path is not None:
    st.subheader("🎥 Video Analysis")
    
    # Create two columns for video and analysis
    col1, col2 = st.columns([1, 1])  # Equal width columns
    
    with col1:
        # Display the annotated video
        st.video(str(st.session_state.annotated_video_path))
        
        # Display frame metrics
        st.subheader("📊 Frame Metrics")
        if st.session_state.metrics:
            metrics_df = pd.DataFrame(st.session_state.metrics)
            st.dataframe(metrics_df)
    
    with col2:
        # Display analysis details
        st.subheader("🔍 Analysis Details")
        
        # Debug information
        st.write("Debug: Checking analysis state")
        st.write(f"Analysis prompt exists: {st.session_state.analysis_prompt is not None}")
        st.write(f"Analysis result exists: {st.session_state.analysis_result is not None}")
        
        if st.session_state.analysis_prompt:
            with st.expander("View Analysis Prompt", expanded=True):
                st.markdown("""
                ### Analysis Prompt
                This is the detailed prompt that will be sent to the AI for analysis:
                """)
                st.markdown("---")
                st.markdown(st.session_state.analysis_prompt)
        
        # Display AI analysis results
        if st.session_state.analysis_result:
            st.subheader("🤖 AI Technique Analysis")
            if "error" in st.session_state.analysis_result and st.session_state.analysis_result["error"]:
                st.error(st.session_state.analysis_result["error"])
            elif st.session_state.analysis_result["analysis"]:
                st.markdown("### Coach's Feedback")
                st.markdown("---")
                st.markdown(st.session_state.analysis_result["analysis"])
            else:
                st.warning("No analysis available. Please check your OpenAI API key configuration.")
        else:
            st.warning("No analysis result available. Please check if the video contains a valid serve.")
