# libraries imported
import os
import re
import cv2
import numpy as np
import face_recognition
from encoding import read_known_names, read_known_files

def main():
    know_users_path = "assets/imgs/users/"
    known_enc = []
    known_names = []
    known_files = []
    face_locations = []

    # train the faces
    read_known_files(know_users_path, known_files)
    read_known_names(know_users_path, known_files, known_names, known_enc)

    # start video capture
    vid_cap = cv2.VideoCapture(0)

    # Continuous loop
    while True:
        
        # read the video capture
        _, frame = vid_cap.read()

        # resize the frame to be proc and convert to rgb
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Find faces in the frame and get encodings
        face_locations = face_recognition.face_locations(rgb_frame)
        shown_enc = face_recognition.face_encodings(rgb_frame, face_locations)

        # compare all face encodings in frame
        for (top, right, bottom, left), face_enc in zip(face_locations, shown_enc):
            matches = face_recognition.compare_faces(known_enc, face_enc)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_enc, face_enc)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_names[best_match_index]

            # Draw a rectangle around each face in frame
            cv2.rectangle(frame, (left, top), (right,bottom), (0,0, 255), 1)

            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), -1)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        # proc_frame = not proc_frame

        '''
        # display results
        for (top, right, bottom, left) in face_locations:
            # top *= 4
            # right *= 4
            # bottom *= 4
            # left *= 4
        
            # Draw a rectangle around teach face in frame
            cv2.rectangle(frame, (left, top), (right,bottom), (0,0, 255), 1)

            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), -1)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        '''

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # end capture and close windows
    vid_cap.release()
    cv2.destroyAllWindows()
    quit()

if __name__ == '__main__':
    main()
    

if __name__ == '__main__':
    main()
    