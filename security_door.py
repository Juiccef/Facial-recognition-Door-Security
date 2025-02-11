import numpy as np
import cv2
import os
import json

# KNN Algorithm
def distance(v1, v2):
    return np.sqrt(((v1 - v2) ** 2).sum())

def knn(train, test, k=5):
    dist = []
    for i in range(train.shape[0]):
        ix = train[i, :-1]
        iy = train[i, -1]
        d = distance(test, ix)
        dist.append([d, iy])
    dk = sorted(dist, key=lambda x: x[0])[:k]
    labels = np.array(dk)[:, -1]
    output = np.unique(labels, return_counts=True)
    index = np.argmax(output[1])
    return output[0][index]

# Load saved doors
settings_file = "door_settings.json"
if os.path.exists(settings_file):
    with open(settings_file, "r") as f:
        doors = json.load(f)
else:
    doors = {}

# Ask user for door setup
if doors:
    print("Existing security doors:")
    for i, door in enumerate(doors.keys(), 1):
        print(f"{i}. {door} (Access: {', '.join(doors[door])})")
    choice = input("\nEnter the door name to use (or type 'new' to create a new one, 'edit' to modify a door): ")
else:
    choice = "new"

if choice == "new":
    door_name = input("Enter the name of the new security door: ")
    authorized_users = [user.strip() for user in input("Enter names of people who have access (comma-separated): ").split(',')]
    authorized_users = [user.strip() for user in authorized_users]
    doors[door_name] = authorized_users  # Save to dictionary
    with open(settings_file, "w") as f:
        json.dump(doors, f)  # Save to file
elif choice == "edit":
    door_name = input("Enter the name of the door to edit: ")
    if door_name in doors:
        print(f"Current authorized users: {', '.join(doors[door_name])}")
        action = input("Do you want to add or remove users? (add/remove): ").strip().lower()
        if action == "add":
            new_users = input("Enter names to add (comma-separated): ").split(',')
            doors[door_name].extend(user.strip() for user in new_users)
        elif action == "remove":
            remove_users = input("Enter names to remove (comma-separated): ").split(',')
            doors[door_name] = [user for user in doors[door_name] if user.strip() not in remove_users]
        with open(settings_file, "w") as f:
            json.dump(doors, f)
        print(f"Updated access list for '{door_name}': {', '.join(doors[door_name])}")
    else:
        print("Door not found. Exiting.")
        exit()
else:
    if choice in doors:
        door_name = choice
        authorized_users = doors[choice]
    else:
        print("Invalid door name. Exiting.")
        exit()

print(f"\nSecurity door '{door_name}' set up! Authorized users: {', '.join(authorized_users)}")

# Setup
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("/Users/youssefgaras/Desktop/Code/Facial recognition/haarcascade_frontalface_alt.xml")

dataset_path = "/Users/youssefgaras/Desktop/Code/Facial recognition/face_dataset/"

face_data = []
labels = []
class_id = 0
names = {}

# Load Dataset
for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        names[class_id] = fx[:-4]
        data_item = np.load(dataset_path + fx)
        face_data.append(data_item)
        target = class_id * np.ones((data_item.shape[0],))
        class_id += 1
        labels.append(target)

# Train Model
if len(face_data) == 0:
    print("No face data found. Please add face data before running the program.")
    exit()

face_dataset = np.concatenate(face_data, axis=0)
face_labels = np.concatenate(labels, axis=0).reshape((-1, 1))
trainset = np.concatenate((face_dataset, face_labels), axis=1)

font = cv2.FONT_HERSHEY_SIMPLEX
last_detected_person = None

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    detected_person = None

    for face in faces:
        x, y, w, h = face
        offset = 5
        face_section = frame[y - offset:y + h + offset, x - offset:x + w + offset]
        face_section = cv2.resize(face_section, (100, 100))

        out = knn(trainset, face_section.flatten())
        detected_person = names[int(out)]

        if detected_person != last_detected_person:
            if detected_person in authorized_users:
                print(f"✅ {detected_person} recognized. Access granted!")
                message = "Access Granted - Door Open"
                color = (0, 255, 0)  # Green
            else:
                print(f"❌ Unauthorized person detected! Access denied.")
                message = "Access Denied"
                color = (0, 0, 255)  # Red
            
            last_detected_person = detected_person

        cv2.putText(frame, message, (x, y - 10), font, 1, color, 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    if detected_person is None:
        last_detected_person = None

    cv2.imshow("Security Door - " + door_name, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
