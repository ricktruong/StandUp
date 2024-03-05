import face_recognition
import cv2
import numpy as np
import time
from datetime import timedelta

""" Global Variables """
# Timer global variables
USER_SITTING_TIME_THRESHOLD = 5                         # Maximum number of seconds the user should be sitting for
FACE_ABSENCE_THRESHOLD = 10                             # Number of Seconds the user should be out of frame for until we stop showing the Stand Up Message
current = time.time()                                   # Current time

# StandUp Notification Algorithm global variables
face_encodings, face_locations, face_names = [], [], [] # Lists of face encodings, face locations, and face names in current frame
last_absence_check_time = None                          # Last time we checked for absence
cumulative_absence_duration = 0.0                       # Total time the user has been absent
stand_command_message = False                           # Flag for issuing "Stand Up!" message
left, top = 0, 0                                        # "Stand Up!" message formatting placeholder variables


""" Helper Functions """
def face_recognition_setup():
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
    # rick_image = face_recognition.load_image_file("known_pictures/rick1.jpg")
    # rick_face_encodings = face_recognition.face_encodings(rick_image)[0]

    # Load image of Daniel and learn to recognize Rick
    daniel_image = face_recognition.load_image_file("known_pictures/daniel.jpg")
    daniel_face_encodings = face_recognition.face_encodings(daniel_image)[0]

    # Known face encodings and labels
    # known_face_encodings = [obama_face_encoding, biden_face_encoding, trump_face_encodings, rick_face_encodings, daniel_face_encodings]
    # known_face_names = ["Barack Obama", "Joe Biden", "Donald Trump", "Rick", "DoctorDothraki"]
    known_face_encodings = [obama_face_encoding, biden_face_encoding, trump_face_encodings, daniel_face_encodings]
    known_face_names = ["Barack Obama", "Joe Biden", "Donald Trump", "DoctorDothraki"]

    return known_face_encodings, known_face_names

def list_camera_devices():
    """ List available camera devices """
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
    
def processing_interval():
    """ OpenCV processing interval """
    global current
    if time.time() - current < 1:
        return False
    current = time.time()
    return True

def add_new_face(known_face_encodings, known_face_names, new_face_encoding, new_face_name):
    """ Append the new face encoding and name to the known faces """
    known_face_encodings.append(new_face_encoding)
    known_face_names.append(new_face_name)

def prompt_for_name(frame, known_face_encodings, known_face_names, face_image):
    """ Prompt for a name and add the new face """
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
    add_new_face(known_face_encodings, known_face_names, new_face_encoding, name)
    
# Search the known faces for a name
def search_faces(frame, known_face_names, known_face_encodings):
    face_names = []
    for index, current_face_encoding in enumerate(face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, current_face_encoding)
        name = "Unknown"

        # If a match was found in known_face_encodings, just use the first one.
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
        
        # If the face is unknown, process it
        else:
            # Find the face location using the current index
            top, right, bottom, left = face_locations[index]
            # Scale the face location since we resized the frame to 1/4 size for faster face recognition processing
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            face_image = frame[top:bottom, left:right]
            prompt_for_name(frame, known_face_encodings, known_face_names, face_image)
            # Break after handling the first unknown face to avoid multiple prompts in a single frame
            break

        face_names.append(name)

    return face_names

# Decide whether or not to tell the user to stand up based on the detection
def detectionDecision():
    global cumulative_absence_duration, last_absence_check_time, stand_command_message
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

# Decide what to display based on stand_command_message
def display(frame):
    global stand_command_message, left, top
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

def standup_notification_algorithm(known_face_encodings, known_face_names):
    global face_encodings, face_locations, face_names, cumulative_absence_duration
    global last_absence_check_time, stand_command_message
    
    """ Webcam OpenCV Functionality """
    # Get a reference to webcam #0 (the default one)
    available_cameras = list_camera_devices()
    if not available_cameras:
        print("No Cameras Found")
        exit()
    video_capture = cv2.VideoCapture(available_cameras[0])
    current_camera_index = 0

    """ StandUp Notification Main Algorithm """
    userSittingTime = 0                             # Number of seconds user has been sitting for
    process_this_frame = True                       # Flag for processing OpenCV frames
    while True:
        # Ensure OpenCV frames are processed every second
        process_this_frame = processing_interval()

        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Only process frames every interval to save processing power
        if process_this_frame:
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])
            
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            """ StandUp Notification Logic """
            # If a user is present / sitting and user is not supposed to be standing, begin incrementing userSittingTime
            if face_encodings and not stand_command_message:
                userSittingTime += 1
            # If a user leaves, reset userSittingTime
            else:
                userSittingTime = 0

            # If user exceeds sitting time, tell user to "Stand Up!". Reset user sitting time.
            if userSittingTime >= USER_SITTING_TIME_THRESHOLD:
                stand_command_message = True
                userSittingTime = 0

            # Search for faces
            face_names = search_faces(frame, known_face_names, known_face_encodings)

            # Decide whether or not to tell the user to stand up based on the time
            detectionDecision()

            print(userSittingTime, USER_SITTING_TIME_THRESHOLD, stand_command_message)
        
        # Manipulate the viewport based on stand_command_message
        display(frame)

        # Check for camera change command
        key = cv2.waitKey(1) & 0xFF
        if key == ord('n'):  # Press 'n' to switch to the next camera
            current_camera_index += 1
            if current_camera_index >= len(available_cameras):
                current_camera_index = 0
            video_capture.release()
            video_capture = cv2.VideoCapture(available_cameras[current_camera_index])
        elif key == ord('q'): # Hit 'q' on the keyboard to quit!
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()

def main():
    # 1. Preload known faces and names into face_recognition
    known_face_encodings, known_face_names = face_recognition_setup()

    # 2. StandUp Notification Algorithm
    standup_notification_algorithm(known_face_encodings, known_face_names)

if __name__ == "__main__":
    main()