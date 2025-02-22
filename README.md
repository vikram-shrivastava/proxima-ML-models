# Flask Backend for Exam Portal with Face and Eye Scanning

## Overview
This Flask-based backend is designed for an online exam portal that incorporates face and eye scanning to ensure remote exam integrity. The system detects and tracks the candidate's face and eye movement using OpenCV and Deep Learning models, preventing cheating during exams.

## Features
- **Face Detection**: Identifies and verifies the candidate's face before and during the exam.
- **Eye Tracking**: Monitors eye movement to detect suspicious activity.
- **Flask API**: Provides RESTful endpoints for real-time detection and monitoring.
- **Live Video Streaming**: Streams the candidate's video feed for continuous monitoring.
- **Alert System**: Triggers alerts if multiple faces are detected or the candidate moves out of frame.

## Technologies Used
- **Flask** - Backend framework for API development
- **OpenCV** - Image processing and face detection
- **Dlib** - Facial landmarks for precise tracking
- **TensorFlow/Keras** - Deep learning models for face verification
- **NumPy & Pandas** - Data handling and processing
- **Socket.IO** - Real-time communication for monitoring

