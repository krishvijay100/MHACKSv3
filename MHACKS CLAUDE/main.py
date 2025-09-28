import streamlit as st
import cv2
import time
import audio_utils
import numpy as np
from scipy.io import wavfile
import json
import os
from datetime import datetime
import transcribe


# Enhanced audio analysis function
def analyze_wav(filename="output.wav"):
    """Analyze audio file for pauses and duration"""
    try:
        sr, y = wavfile.read(filename)

        if y.ndim > 1:  # convert to mono if stereo
            y = y.mean(axis=1)

        y = y.astype(np.float32)
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))  # normalize to [-1, 1]

        pauses = 0
        threshold = 0.2
        min_pause_len = int(0.1 * sr)  # 0.1s tolerance

        is_silent = np.abs(y) < threshold
        count = 0

        for val in is_silent:
            if val:
                count += 1
            else:
                if count >= min_pause_len:
                    pauses += 1
                count = 0

        # Check for final pause
        if count >= min_pause_len:
            pauses += 1

        duration = len(y) / sr

        return {
            'pauses': pauses,
            'duration': duration,
            'sample_rate': sr,
            'total_samples': len(y)
        }
    except Exception as e:
        print(f"Error analyzing audio: {e}")
        return {
            'pauses': 0,
            'duration': 0,
            'sample_rate': 0,
            'total_samples': 0
        }


# Hand gesture detection using OpenCV
def detect_hand_gestures(frame):
    """Simple hand gesture detection using contour analysis"""
    try:
        # Convert to HSV for better skin detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define range for skin color (this is a simple approximation)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)

        # Create mask for skin color
        mask = cv2.inRange(hsv, lower_skin, upper_skin)

        # Apply morphological operations to reduce noise
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours by area (potential hands)
        hand_contours = []
        min_area = 1000  # Minimum area for a hand
        max_area = 50000  # Maximum area for a hand

        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                # Check if contour is roughly hand-shaped using aspect ratio
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / float(h)
                if 0.3 < aspect_ratio < 3.0:  # Reasonable aspect ratio for hands
                    hand_contours.append(contour)

        return len(hand_contours)

    except Exception as e:
        print(f"Error in hand gesture detection: {e}")
        return 0


st.set_page_config(
    page_title="Advanced Speechtomizor",
    page_icon="🎤",
    layout="wide"
)

st.title("🎤 Advanced Speechtomizor")
st.markdown("**Comprehensive Speech, Eye Contact, and Gesture Analysis**")

# Initialize face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# Initialize session state
if "run" not in st.session_state:
    st.session_state.run = False
if "audio_recorder" not in st.session_state:
    st.session_state.audio_recorder = None
if "recording_data" not in st.session_state:
    st.session_state.recording_data = {}
if "gesture_count" not in st.session_state:
    st.session_state.gesture_count = 0
if "transcript_text" not in st.session_state:
    st.session_state.transcript_text = ""

# Create layout
col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

with col1:
    start_button = st.button("🔴 Start Recording", use_container_width=True)

with col2:
    stop_button = st.button("⏹️ Stop Recording", use_container_width=True)

with col3:
    analyze_button = st.button("🔍 Analyze Results", use_container_width=True)

with col4:
    reset_button = st.button("🔄 Reset", use_container_width=True)

status_placeholder = st.empty()

# Handle start button
if start_button and not st.session_state.run:
    st.session_state.run = True

    # Create new audio recorder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    audio_filename = f"recording_{timestamp}.wav"
    st.session_state.audio_recorder = audio_utils.create_recorder(audio_filename)

    # Start audio recording
    if audio_utils.start_recording(st.session_state.audio_recorder):
        status_placeholder.success("🎙️ Recording started! Audio, video, and gesture analysis active.")

        # Reset all counters
        st.session_state.start_time = time.time()
        st.session_state.total_time = 0
        st.session_state.eye_time = 0
        st.session_state.gesture_count = 0
        st.session_state.last_check = st.session_state.start_time
        st.session_state.gesture_frames = []  # Track gestures per frame
    else:
        st.session_state.run = False
        status_placeholder.error("❌ Failed to start audio recording!")

# Handle stop button
if stop_button and st.session_state.run:
    st.session_state.run = False

    # Stop audio recording
    if st.session_state.audio_recorder:
        audio_info = audio_utils.stop_recording(st.session_state.audio_recorder)
        if audio_info:
            st.session_state.recording_data['audio_info'] = audio_info

            # Analyze the audio file
            audio_analysis = analyze_wav(audio_info['filename'])
            st.session_state.recording_data['audio_analysis'] = audio_analysis

            status_placeholder.success(f"✅ Recording stopped! Audio saved as: {audio_info['filename']}")
        else:
            status_placeholder.warning("⚠️ Recording stopped, but audio save failed!")

    # Calculate final metrics
    if 'total_time' in st.session_state and st.session_state.total_time > 0:
        eye_contact_ratio = st.session_state.eye_time / st.session_state.total_time

        # Calculate average gestures per minute
        gestures_per_minute = (
                                          st.session_state.gesture_count / st.session_state.total_time) * 60 if st.session_state.total_time > 0 else 0

        st.session_state.recording_data.update({
            'total_time': st.session_state.total_time,
            'eye_time': st.session_state.eye_time,
            'eye_contact_ratio': eye_contact_ratio,
            'gesture_count': st.session_state.gesture_count,
            'gestures_per_minute': gestures_per_minute
        })

# Handle analyze button
if analyze_button and 'audio_analysis' in st.session_state.recording_data:
    try:
        # Get transcript input from user
        transcript_input = st.text_area(
            "📝 Enter your speech transcript for analysis:",
            value=st.session_state.transcript_text,
            height=100,
            help="Paste or type the transcript of what you said during the recording"
        )

        if transcript_input.strip():
            st.session_state.transcript_text = transcript_input.strip()

            # Prepare data for analysis
            audio_analysis = st.session_state.recording_data['audio_analysis']
            eye_contact_ratio = st.session_state.recording_data.get('eye_contact_ratio', 0)
            gesture_count = st.session_state.recording_data.get('gesture_count', 0)

            # Generate feedback using transcribe module
            with st.spinner("🤖 Analyzing your speech performance..."):
                analysis_result = transcribe.generate_feedback(
                    transcript=st.session_state.transcript_text,
                    pauses=audio_analysis['pauses'],
                    eyecontactratio=eye_contact_ratio,
                    handgestures=gesture_count
                )

            # Store results for results page
            st.session_state.analysis_data = analysis_result
            st.session_state.analysis_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.pause_count = audio_analysis['pauses']
            st.session_state.speech_duration = f"{audio_analysis['duration']:.1f}s"

            st.success("🎉 Analysis complete! Check the results page for detailed feedback.")

            # Display quick results preview
            if 'scores' in analysis_result:
                scores = analysis_result['scores']
                if len(scores) >= 4:
                    st.info(
                        f"📊 Quick Preview - Eye Contact: {scores[0]}/100, Fluency: {scores[1]}/100, Vocabulary: {scores[2]}/100, Overall: {scores[3]}/100")
        else:
            st.warning("⚠️ Please enter a transcript to analyze your speech.")

    except Exception as e:
        st.error(f"❌ Error during analysis: {str(e)}")

# Handle reset button
if reset_button:
    # Reset all session state
    for key in list(st.session_state.keys()):
        if key not in ['analysis_data', 'analysis_date']:  # Keep analysis results
            del st.session_state[key]
    st.session_state.run = False
    st.session_state.recording_data = {}
    st.session_state.gesture_count = 0
    st.session_state.transcript_text = ""
    status_placeholder.info("🔄 Application reset successfully!")

# Create placeholders for video and stats
video_col, stats_col = st.columns([2, 1])

with video_col:
    st.subheader("📹 Live Analysis Feed")
    video_placeholder = st.empty()

with stats_col:
    st.subheader("📊 Real-time Metrics")
    stats_placeholder = st.empty()

# Main video processing loop
if st.session_state.run:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("❌ Could not open camera. Please check your camera permissions.")
        st.session_state.run = False
    else:
        # Initialize timing variables
        if 'start_time' not in st.session_state:
            st.session_state.start_time = time.time()
            st.session_state.last_check = st.session_state.start_time
            st.session_state.total_time = 0
            st.session_state.eye_time = 0
            st.session_state.gesture_count = 0

        frame_count = 0
        gesture_check_interval = 10  # Check gestures every 10 frames for performance

        while st.session_state.run:
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Failed to grab frame from camera.")
                break

            frame_count += 1

            # Update timing
            now = time.time()
            elapsed = now - st.session_state.last_check
            st.session_state.total_time += elapsed
            st.session_state.last_check = now

            # Process frame for face and eye detection
            raw_frame = frame.copy()
            img = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
            eyes_detected = False
            face_count = 0

            # Detect faces
            faces = face_cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=5)
            face_count = len(faces)

            for (x, y, w, h) in faces:
                # Draw face rectangle
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Detect eyes within face region
                roi_gray = img[y:y + h, x:x + w]
                eyes = eye_cascade.detectMultiScale(roi_gray)

                if len(eyes) >= 2:
                    eyes_detected = True
                    # Draw circles around eyes
                    for (ex, ey, ew, eh) in eyes[:2]:
                        center = (x + ex + ew // 2, y + ey + eh // 2)
                        radius = int(round((ew + eh) * 0.25))
                        cv2.circle(frame, center, radius, (255, 0, 0), 2)

            # Update eye contact time
            if eyes_detected:
                st.session_state.eye_time += elapsed

            # Detect hand gestures (check every few frames for performance)
            if frame_count % gesture_check_interval == 0:
                current_gestures = detect_hand_gestures(raw_frame)
                if current_gestures > 0:
                    st.session_state.gesture_count += current_gestures

            # Add overlay information to frame
            overlay_y = 30
            line_height = 25

            cv2.putText(raw_frame, f"Recording: {st.session_state.total_time:.1f}s",
                        (10, overlay_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            overlay_y += line_height

            eye_status = "Eyes: Detected" if eyes_detected else "Eyes: Not Detected"
            cv2.putText(raw_frame, eye_status, (10, overlay_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0) if eyes_detected else (0, 0, 255), 2)
            overlay_y += line_height

            cv2.putText(raw_frame, f"Gestures: {st.session_state.gesture_count}",
                        (10, overlay_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

            # Display the frame
            raw_rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(raw_rgb, channels="RGB", use_column_width=True)

            # Update statistics
            if st.session_state.total_time > 0:
                eye_contact_percentage = (st.session_state.eye_time / st.session_state.total_time) * 100
                gestures_per_minute = (st.session_state.gesture_count / st.session_state.total_time) * 60

                # Get audio recording duration
                audio_duration = 0
                if st.session_state.audio_recorder:
                    audio_duration = audio_utils.get_duration(st.session_state.audio_recorder)

                stats_html = f"""
                <div style="padding: 1rem; background-color: #f0f2f6; border-radius: 0.5rem; margin-bottom: 1rem;">
                    <h4>📈 Live Performance Metrics</h4>
                    <div style="display: grid; gap: 0.5rem;">
                        <p><strong>⏱️ Total Time:</strong> {st.session_state.total_time:.1f}s</p>
                        <p><strong>👁️ Eye Contact:</strong> {st.session_state.eye_time:.1f}s ({eye_contact_percentage:.1f}%)</p>
                        <p><strong>🎙️ Audio Duration:</strong> {audio_duration:.1f}s</p>
                        <p><strong>👋 Total Gestures:</strong> {st.session_state.gesture_count}</p>
                        <p><strong>📊 Gestures/min:</strong> {gestures_per_minute:.1f}</p>
                        <p><strong>👤 Faces:</strong> {face_count}</p>
                    </div>
                </div>
                """

                # Performance indicators
                eye_indicator = "🟢" if eye_contact_percentage > 60 else "🟡" if eye_contact_percentage > 30 else "🔴"
                gesture_indicator = "🟢" if 10 < gestures_per_minute < 60 else "🟡" if gestures_per_minute > 0 else "🔴"

                indicators_html = f"""
                <div style="padding: 1rem; background-color: #e1f5fe; border-radius: 0.5rem;">
                    <h4>🚦 Performance Indicators</h4>
                    <p>{eye_indicator} <strong>Eye Contact:</strong> {'Excellent' if eye_contact_percentage > 60 else 'Fair' if eye_contact_percentage > 30 else 'Needs Work'}</p>
                    <p>{gesture_indicator} <strong>Gestures:</strong> {'Good' if 10 < gestures_per_minute < 60 else 'Active' if gestures_per_minute > 0 else 'Static'}</p>
                </div>
                """

                stats_placeholder.markdown(stats_html + indicators_html, unsafe_allow_html=True)

            # Small delay to prevent overwhelming the UI
            time.sleep(0.033)  # ~30 FPS

        # Clean up camera
        cap.release()
        cv2.destroyAllWindows()

# Display recording summary if available
if not st.session_state.run and st.session_state.recording_data:
    st.markdown("---")
    st.subheader("📋 Recording Summary")

    data = st.session_state.recording_data

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if 'total_time' in data:
            st.metric("Total Duration", f"{data['total_time']:.1f}s")

    with col2:
        if 'eye_contact_ratio' in data:
            st.metric("Eye Contact Ratio", f"{data['eye_contact_ratio'] * 100:.1f}%")

    with col3:
        if 'gesture_count' in data:
            st.metric("Total Gestures", data['gesture_count'])

    with col4:
        if 'audio_analysis' in data:
            st.metric("Detected Pauses", data['audio_analysis']['pauses'])

    # Show detailed audio analysis
    if 'audio_analysis' in data:
        audio_info = data['audio_analysis']
        st.info(f"""
        🎙️ **Audio Analysis:**
        - Duration: {audio_info['duration']:.2f} seconds
        - Pauses detected: {audio_info['pauses']}
        - Sample rate: {audio_info['sample_rate']} Hz
        - Total samples: {audio_info['total_samples']:,}
        """)

# Instructions
with st.expander("ℹ️ How to Use Advanced Speechtomizor"):
    st.markdown("""
    ## 🎯 Complete Speech Analysis Workflow:

    ### 1. **Recording Phase**
    - Click **'Start Recording'** to begin comprehensive analysis
    - Speak naturally while looking at the camera
    - Use natural hand gestures while presenting
    - Watch real-time performance indicators

    ### 2. **Analysis Phase**
    - Click **'Stop Recording'** when finished
    - Enter your speech transcript in the text area
    - Click **'Analyze Results'** for AI-powered feedback

    ### 3. **Review Phase**
    - Check your performance metrics and scores
    - Review personalized recommendations
    - Use insights to improve your presentation skills

    ## 📊 What We Analyze:
    - **👁️ Eye Contact:** Percentage of time maintaining camera contact
    - **🎙️ Speech Fluency:** Pauses, filler words, and flow analysis
    - **👋 Hand Gestures:** Natural gesture frequency and timing
    - **📚 Vocabulary:** Word choice and language complexity
    - **🏆 Overall Performance:** Comprehensive presentation score

    ## 💡 Tips for Best Results:
    - **Lighting:** Ensure good lighting for accurate face detection
    - **Positioning:** Stay centered in the camera frame
    - **Audio:** Speak clearly into your microphone
    - **Natural Behavior:** Act as if presenting to a real audience
    - **Transcript Accuracy:** Provide accurate transcript for best analysis
    """)

# Footer
st.markdown("---")
st.markdown(
    "🚀 **Advanced Speechtomizor** - AI-Powered Presentation Analysis | Built with Streamlit, OpenCV, and Machine Learning")