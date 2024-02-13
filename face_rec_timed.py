import face_recognition
import cv2
import numpy as np
import time

# Both measured in seconds
CHECK_INTERVAL = 15
STAND_INTERVAL = 60 * 1

# We don't want to tell them to stand up more often than we check
if CHECK_INTERVAL >= STAND_INTERVAL:
    exit()

start = time.time()
current = time.time()

measureSecond1 = time.time()
time.sleep(1)
measureSecone2 = time.time()
print(measureSecone2 - measureSecond1)

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

# Known face encodings and labels
known_face_encodings = [obama_face_encoding, biden_face_encoding, trump_face_encodings, rick_face_encodings]
known_face_names = ["Barack Obama", "Joe Biden", "Donald Trump", "Me Myself and I"]


""" Webcam OpenCV Functionality """
# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
stand_command = False

while True:
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

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    # process_this_frame = not process_this_frame

    if stand_command:
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

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Check if there is a good time to stand up
    values = interval()

    process_this_frame = values[0]
    stand_command = values[1]

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()