import cv2
import time
import requests 

# --- Configuration ---
CLOUD_FUNCTION_URL = "http://20.57.11.23:8000/ingest-alert/"
API_KEY = "your-secret-key-here-12345"

MISSING_TIMEOUT_SECONDS = 5
MULTIPLE_ALERT_COOLDOWN = 5.0 # Your logic: 5 sec cooldown

# Cooldown for the "Normal" state to fix flicker
NORMAL_RESET_COOLDOWN = 3.0  # 3 seconds of "Normal" to reset alerts
# -------------------------------------------------------------

# --- Cloud Upload Function (Stays the same) ---
def send_alert_to_cloud(alert_type, incident_frame):
    """
    Sends a single image snapshot and an alert type to the cloud.
    """
    print(f"CLD: Attempting to send alert: {alert_type}")
    
    try:
        ret, buffer = cv2.imencode('.jpg', incident_frame)
        if not ret:
            print("ERR: Could not encode image frame.")
            return

        files = { 'image': ('incident.jpg', buffer.tobytes(), 'image/jpeg') }
        payload = { 'alert_type': alert_type, 'timestamp': time.time() }
        headers = { 'X-API-Key': API_KEY }

        response = requests.post(
            CLOUD_FUNCTION_URL, 
            files=files, 
            data=payload,
            headers=headers,
            timeout=10
        )
        print(f"CLD: Upload success! Server responded with: {response.status_code}")
        print(f"CLD: Server message: {response.json()}")

    except requests.exceptions.ConnectionError:
        print(f"ERR: Connection refused. Is the cloud server running at {CLOUD_FUNCTION_URL}?")
    except Exception as e:
        print(f"ERR: An unknown error occurred during upload: {e}")

# --- Main Application Logic ---

# 1. Load the pre-trained Haar Cascade model
try:
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        print("Error: Could not load Haar Cascade file.")
        exit()
except Exception as e:
    print(f"An error occurred loading the cascade file: {e}")
    exit()

# 2. Initialize video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open webcam.")
    exit()

print("Starting Edge Proctor (FINAL Robust Logic)... Press 'q' to quit.")

# --- 2. State Variables (Robust Version) ---
missing_start_time = None
alert_sent_for_missing = False # Tracks the "Missing" alert state
last_multiple_alert_time = 0.0 # Tracks the "Multiple" alert time
normal_start_time = None       # Timer for the "Normal" state cooldown
# ---------------------------------------------------------

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    face_count = len(faces)

    alert_message = "Normal"
    alert_color = (0, 255, 0) # Green

    # 4. Implement Robust, Stateful Logic
    
    if face_count == 0:
        # --- Student Missing Logic ---
        alert_message = "ALERT: Student Missing!"
        alert_color = (0, 255, 255) # Yellow
        
        # We are not normal, so reset the "normal" timer
        normal_start_time = None 

        if missing_start_time is None:
            # This will now only print ONCE per event
            print("LOG: Student missing, starting timer...")
            missing_start_time = time.time()
        else:
            elapsed_time = time.time() - missing_start_time
            if elapsed_time > MISSING_TIMEOUT_SECONDS and not alert_sent_for_missing:
                print(f"LOG: Student missing for {elapsed_time:.1f}s. Sending alert!")
                send_alert_to_cloud("STUDENT_MISSING", frame)
                alert_sent_for_missing = True
        
    elif face_count > 1:
        # --- Multiple People Logic (Your Timed Cooldown) ---
        alert_message = "ALERT: Multiple People Detected!"
        alert_color = (0, 0, 255) # Red
        
        # We are not normal, so reset the "normal" timer
        normal_start_time = None

        current_time = time.time()
        if (current_time - last_multiple_alert_time) > MULTIPLE_ALERT_COOLDOWN:
            print("LOG: Multiple people detected. Sending alert!")
            send_alert_to_cloud("MULTIPLE_PEOPLE", frame)
            last_multiple_alert_time = current_time
        
        # We DO NOT reset the 'missing' timer here. 
        # The 'Normal' state will handle the full reset when the face returns.

    else:
        # --- Normal Logic (face_count == 1) [THE FIX] ---
        alert_message = "Normal"
        alert_color = (0, 255, 0) # Green
        
        # This is the ONLY place the "missing" timer should be reset.
        missing_start_time = None
        
        # This is the FIX: Only reset the "missing" alert flag
        # after we've been "Normal" for 3+ seconds.
        if alert_sent_for_missing:
            if normal_start_time is None:
                print("LOG: Normal state detected. Starting reset cooldown timer...")
                normal_start_time = time.time()
            else:
                elapsed_normal_time = time.time() - normal_start_time
                if elapsed_normal_time > NORMAL_RESET_COOLDOWN:
                    print(f"LOG: Normal for {elapsed_normal_time:.1f}s. Resetting 'Missing' alert.")
                    alert_sent_for_missing = False
                    normal_start_time = None
        else:
            # No "missing" alert is active, so no need for the "normal" timer
            normal_start_time = None
            
    # 5. Draw rectangles and text
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), alert_color, 2)
    cv2.putText(
        frame, alert_message, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, alert_color, 2
    )

    # 6. Display the resulting frame
    cv2.imshow('Edge Proctor - Press Q to Quit', frame)

    # 7. Check for 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 8. Cleanup
print("Shutting down...")
cap.release()
cv2.destroyAllWindows()