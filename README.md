Facial Recognition Door Security
Welcome to my facial recognition security door project! Please follow the instructions below to ensure proper functionality.

Overview
This project implements a facial recognition-based security system using OpenCV and K-Nearest Neighbors (KNN). It allows users to set up security doors, manage access permissions, and authenticate individuals in real-time. The system detects faces via a webcam, classifies them using a trained dataset, and grants or denies access based on stored authorization data. Unauthorized attempts trigger alerts, enhancing security.

Setup & Usage
Step 1: Verify Camera Functionality
Run open_cam.py in your IDE to ensure your device’s camera is working properly.

Step 2: Train the Facial Recognition Model
Run face_data.py to capture and store images of your face. Keep it running until at least 10 images are collected, moving your face slightly from side to side for better accuracy.

Step 3: Test Facial Recognition
Run facial_recognition.py to check if the program correctly detects and labels your face with a tracking box.

Step 4: Configure the Security Door
Run security_door.py and follow the instructions to:

Add a new security door.
Assign authorized users (note: only faces previously added in face_data.py can be authorized).
Once set up, the system will recognize authorized users and grant or deny access accordingly.

Enjoy!
I hope you find this project useful. Feel free to explore and improve it!
