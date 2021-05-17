import face_recognition as fr
import cv2 as cv

l = []
video_capture = cv.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
deepak_image = fr.load_image_file("E:\\deepak\\Miniprojects\\Miniproject\\assets\\deepak.jpg")
deepak_face_encoding = fr.face_encodings(deepak_image)[0]

# Load a second sample picture and learn how to recognize it.
payal_image = fr.load_image_file("E:\\deepak\\Miniprojects\\Miniproject\\assets\\payal.jpeg")
payal_face_encoding = fr.face_encodings(payal_image)[0]

yash_image = fr.load_image_file("E:\\deepak\\Miniprojects\\Miniproject\\assets\\yash.jpeg")
yash_face_encoding = fr.face_encodings(yash_image)[0]

# Creating list of known face encodings and their names
known_face_encodings = [
    deepak_face_encoding,
    payal_face_encoding,
    yash_face_encoding
]
known_face_names = [
    "Deepak",
    "Payal",
    "Yash",

]

# Initialize some variables for proccessing every frame
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which frame uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = fr.face_locations(rgb_small_frame)
        face_encodings = fr.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = fr.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
            name = "Unknown"

            # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
                l.append(name)



            face_names.append(name)

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), -1)
        font = cv.FONT_HERSHEY_TRIPLEX
        cv.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv.destroyAllWindows()

l = (set(l))
fi = open('E:\\deepak\\Miniprojects\\Miniproject\\sample.txt','w')
for x in l:
    fi.write(x+'\n')
fi.close()