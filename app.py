from flask import Flask
from flask_socketio import SocketIO
from flask_cors import CORS
import cv2
import base64
import eventlet

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

def generate_frames():
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        focus_status = "Not Focusing!"

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(eyes) >= 2: 
                focus_status = "Candidate is Focusing!"
                cv2.putText(frame, "Focusing", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:  
                cv2.putText(frame, "Not Focusing!", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_data = base64.b64encode(buffer).decode('utf-8')

        socketio.emit("video_feed", {"image": frame_data, "focus_status": focus_status})

        eventlet.sleep(0.03)  
    cap.release()

@app.route("/")
def index():
    return "WebSocket Video Streaming Server with Face Tracking"

@socketio.on("start_video")
def start_video():
    generate_frames()

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=8081, allow_unsafe_werkzeug=True)
