import cv2
import streamlit as st
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# 1. Load the pre-trained Haar Cascade model for face detection
# Ensure this file is in the same directory as your script
try:
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        st.error("Error: Could not load Haar Cascade file. Make sure 'haarcascade_frontalface_default.xml' is in the same directory.")
        st.stop()
except Exception as e:
    st.error(f"An error occurred loading the cascade file: {e}")
    st.stop()


# 2. Define the class that will process the video frames
class FaceProctorTransformer(VideoTransformerBase):
    def __init__(self):
        self.face_count = 0
        self.alert_message = "Normal"
        self.alert_type = "success"

    def transform(self, frame):
        # Convert the frame to a format OpenCV can use
        img = frame.to_ndarray(format="bgr24")
        
        # Convert to grayscale for the face detector (it's more efficient)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 3. Detect faces in the grayscale image
        # 'scaleFactor' and 'minNeighbors' are tuning parameters.
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        self.face_count = len(faces)

        # 4. Implement the Edge Logic and draw on the image
        alert_color = (0, 255, 0) # Green for "Normal"

        if self.face_count == 0:
            self.alert_message = "ALERT: Student Missing!"
            self.alert_type = "warning"
            alert_color = (0, 255, 255) # Yellow
            
        elif self.face_count > 1:
            self.alert_message = "ALERT: Multiple People Detected!"
            self.alert_type = "error"
            alert_color = (0, 0, 255) # Red
            
        else:
            self.alert_message = "Normal"
            self.alert_type = "success"
            alert_color = (0, 255, 0) # Green

        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), alert_color, 2)
            
        # Add the alert text to the video frame itself
        cv2.putText(
            img,
            self.alert_message,
            (10, 30), # Position
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, # Font scale
            alert_color,
            2 # Thickness
        )

        return img

# --- Streamlit App UI ---

st.set_page_config(page_title="Edge Proctor", layout="centered")
st.title("👨‍🏫 Live Edge Proctoring System")
st.write("This app runs locally on your machine. It processes your webcam feed in real-time to count faces and detect anomalies.")

# We use streamlit_webrtc to get the live video feed
# It will handle the webcam access and display the processed frames
# --- Alternative Streamlit-WebRTC Configuration ---

ctx = webrtc_streamer(
    key="proctor",
    video_transformer_factory=FaceProctorTransformer,
    # This configuration is minimal, hoping to force a direct local connection
    rtc_configuration=RTCConfiguration(
        {"iceServers": []} # Use an empty list to disable external STUN servers
    ), 
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# Display the status message below the video
if ctx.video_transformer:
    st.markdown(f"### <strong>Current Status:</strong>", unsafe_allow_html=True)
    
    # Use st.empty() to create a placeholder that we can update
    status_placeholder = st.empty()

    while True:
        # We need to access the transformer object to get the latest status
        if ctx.video_transformer:
            message = ctx.video_transformer.alert_message
            alert_type = ctx.video_transformer.alert_type

            # Update the placeholder with the correct alert type
            if alert_type == "success":
                status_placeholder.success(message)
            elif alert_type == "warning":
                status_placeholder.warning(message)
            elif alert_type == "error":
                status_placeholder.error(message)
        
        # A short sleep to prevent the loop from running too fast
        cv2.waitKey(100)

st.info("How it works: The app uses OpenCV to detect faces. If 0 or >1 faces are seen, it triggers an alert. All processing happens on *your* device (the 'edge').")