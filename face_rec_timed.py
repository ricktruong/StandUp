import face_recognition
import cv2
import numpy as np
import time
from datetime import timedelta

# Both measured in seconds
CHECK_INTERVAL = 3
STAND_INTERVAL = 10 * 1
FACE_ABSENCE_THRESHOLD = 15  # Number of Seconds the user should be out of frame for until we stop showing the Stand Up Message

# We don't want to tell them to stand up more often than we check
if CHECK_INTERVAL >= STAND_INTERVAL:
    exit()

start = time.time()
current = time.time()


def add_new_face(new_face_encoding, new_face_name):
    # Append the new face encoding and name to the known faces
    known_face_encodings.append(new_face_encoding)
    known_face_names.append(new_face_name)

# Function to prompt for a name and add the new face
def prompt_for_name(face_image):
    # Display the face image in the window
    cv2.imshow('Video', frame)
    cv2.waitKey(1) #Pause
    
    # Prompt the user for a name for the unrecognized face
    print("An unrecognized face was detected.")
    name = input("Please enter a name for the new face: ")
    
    # Learn the new face by encoding it
    rgb_face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    new_face_encoding = face_recognition.face_encodings(rgb_face_image)[0]

    # Add the new face to the known faces
    add_new_face(new_face_encoding, name)
    return name
    

def list_camera_devices():
    # checks the first 10 indexes.
    index = 0
    arr = []
    i = 10
    while i > 0:
        cap = cv2.VideoCapture(index)
        if cap.read()[0]:
            arr.append(index)
            cap.release()
        index += 1
        i -= 1
    return arr
def change_camera(camera_index):
    global video_capture
    video_capture.release()
    video_capture = cv2.VideoCapture(camera_index)
def interval():
    global start
    global current
    # Don't duplicate things
    if time.time() - current > 1:
        current = time.time()
    else:
        return [False, False]
    
    # Has 15 minutes passed?
    if (current - start) >= STAND_INTERVAL:
        start = time.time()
        return [True, True]

    # Has a 30 second interval passed?
    if (current - start) >= CHECK_INTERVAL:
        current = time.time()
        return [True, False]
    else:
        return [False, False]

""" face_recognition Setup """
# Load image of Obama and learn to recognize Obama
obama_image = face_recognition.load_image_file("known_pictures/obama.jpg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

# Load image of Biden and learn to recognize Biden
biden_image = face_recognition.load_image_file("known_pictures/biden.jpg")
biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

# Load image of Trump and learn to recognize Trump
trump_image = face_recognition.load_image_file("known_pictures/trump.jpg")
trump_face_encodings = face_recognition.face_encodings(trump_image)[0]

# Load image of Rick and learn to recognize Rick
rick_image = face_recognition.load_image_file("known_pictures/rick1.jpg")
rick_face_encodings = face_recognition.face_encodings(rick_image)[0]

# Load image of Daniel and learn to recognize Rick
daniel_image = face_recognition.load_image_file("known_pictures/daniel.jpg")
daniel_face_encodings = face_recognition.face_encodings(daniel_image)[0]

# Known face encodings and labels
known_face_encodings = [obama_face_encoding, biden_face_encoding, trump_face_encodings, rick_face_encodings, daniel_face_encodings]
known_face_names = ["Barack Obama", "Joe Biden", "Donald Trump", "Me Myself and I", "DoctorDothraki"]


""" Webcam OpenCV Functionality """
# Get a reference to webcam #0 (the default one)
available_cameras = list_camera_devices()
if not available_cameras:
    print("No Cameras Found")
    exit()
video_capture = cv2.VideoCapture(available_cameras[0])
current_camera_index = 0


# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
stand_command = False
stand_command_message = False
face_absent_start = None  # Timestamp of when a face was last seen
stand_command_message_start_time = None
cumulative_absence_duration = 0.0  # Total time the user has been absent
last_absence_check_time = None  # Last time we checked for absence
left = 0
top = 0
while True:
    # Check if there is a good time to stand up
    values = interval()
    process_this_frame = values[0]
    stand_command = values[1]

    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Only process every other frame of video to save time
    if process_this_frame:
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])
        
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            if name == "Unknown":
                # Crop the face from the frame
                top, right, bottom, left = [v * 4 for v in face_locations[face_encodings.index(face_encoding)]]
                face_image = frame[top:bottom, left:right]
                name = prompt_for_name(face_image)

            face_names.append(name)

        
    current_time = time.time()
    if face_names:  # If any face is detected
        # User is present; pause the timer by not updating the cumulative_absence_duration
        if last_absence_check_time is not None:
            last_absence_check_time = current_time  # Update the last check time to now
    elif stand_command_message:
        # No known face detected; check if this is the first time we're noticing they're gone
        if last_absence_check_time is None:
            last_absence_check_time = current_time
        else:
            # Calculate how long it's been since we last updated the absence duration
            time_since_last_check = current_time - last_absence_check_time
            cumulative_absence_duration += time_since_last_check
            last_absence_check_time = current_time

            # Check if the cumulative absence duration exceeds the threshold
            if cumulative_absence_duration > FACE_ABSENCE_THRESHOLD:
                stand_command_message = False
                cumulative_absence_duration = 0.0  # Reset the absence duration as user is present and has been notified
                last_absence_check_time = None

    # process_this_frame = not process_this_frame

    if stand_command:
        stand_command_message = True
        stand_command_message_start_time = time.time()
        stand_command = False
        print("Stand up")
    
    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        if stand_command_message:
            font = cv2.FONT_HERSHEY_SIMPLEX
            message = "Stand up!"
            cv2.putText(frame, message, (left + 15, top - 20), font, 1.0, (0, 255, 0), 2)
        
    if stand_command_message:
        font = cv2.FONT_HERSHEY_SIMPLEX
        remaining_time = max(FACE_ABSENCE_THRESHOLD - cumulative_absence_duration, 0)
        remaining_time_delta = timedelta(seconds=remaining_time)
        minutes, seconds = divmod(remaining_time_delta.seconds, 60)
        formatted_time = f"{minutes:02d}:{seconds:02d}"
        message = f"Time Remaining: {formatted_time}!"
        cv2.putText(frame, message, (left-80, top - 50), font, 1.0, (252, 186, 3), 2)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Check for camera change command
    key = cv2.waitKey(1) & 0xFF
    if key == ord('n'):  # Press 'n' to switch to the next camera
        current_camera_index += 1
        if current_camera_index >= len(available_cameras):
            current_camera_index = 0
        change_camera(available_cameras[current_camera_index])
    elif key == ord('q'): # Hit 'q' on the keyboard to quit!
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()