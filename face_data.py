import cv2
import numpy as np

# Open webcam
cap = cv2.VideoCapture(0)  # 0 = default webcam
face_cascade = cv2.CascadeClassifier("/Users/youssefgaras/Desktop/Code/Facial recognition/haarcascade_frontalface_alt.xml")

skip = 0
face_data = []
dataset_path = "/Users/youssefgaras/Desktop/Code/Facial recognition/face_dataset/"

# Take input for the person's name
file_name = input("Enter the person's name: ")

while True:
    ret, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if not ret:
        continue

    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)  # Detect faces
    if len(faces) == 0:
        continue

    k = 1
    # Sort faces by area (descending)
    faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)

    skip += 1
    for face in faces[:1]:  # Process the largest face only
        x, y, w, h = face
        offset = 5
        face_offset = frame[y - offset:y + h + offset, x - offset:x + w + offset]
        face_selection = cv2.resize(face_offset, (100, 100))

        if skip % 10 == 0:
            face_data.append(face_selection)
            print(f"Captured: {len(face_data)} images")

            # Display the captured face
            cv2.imshow("Captured Face", face_selection)

            # Draw a rectangle around the detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show frame with detected face
    cv2.imshow("Frame", frame)

    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

# Convert face data to numpy array and save
face_data = np.array(face_data)
face_data = face_data.reshape((face_data.shape[0], -1))  # Flatten the images
print(f"Dataset shape: {face_data.shape}")

# Save the dataset
np.save(dataset_path + file_name, face_data)
print(f"Dataset saved at: {dataset_path + file_name}.npy")

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
