# libraries imported
import os
import re
import cv2
import numpy as np
import face_recognition
from encoding import read_known_names, read_known_files

def main():
    known_users_path = "assets/imgs/users/"
    input_path = "assets/imgs/inputs/"
    known_enc = []
    known_names = []
    known_files = []

    # test image
    input = "user-one.jpg"

    # train the faces
    read_known_files(known_users_path, known_files)
    read_known_names(known_users_path, known_files, known_names, known_enc)

    # live scan mode
    # live_scan(known_enc, known_names)

    # input mode
    scan_input(known_enc, known_names, input_path, input)

    quit()

def live_scan(known_enc, known_names):

    # face locations array
    face_locations=[]

    # start video capture
    vid_cap = cv2.VideoCapture(0)

    # make a named window
    cv2.namedWindow('Live Scan', cv2.WINDOW_AUTOSIZE)

    # Continuous loop
    while cv2.getWindowProperty('Live Scan', 0) >= 0:
        
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

            # Check if known or unknown
            if name is "Unknown":
                # Draw a rectangle around each face in frame
                cv2.rectangle(frame, (left, top), (right,bottom), (0,0, 255), 1)

                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), -1)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
            else:
                # Draw a rectangle around each face in frame
                cv2.rectangle(frame, (left, top), (right,bottom), (0, 128, 0), 1)

                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 128, 0), -1)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Live Scan', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # end capture and close windows
    vid_cap.release()
    cv2.destroyAllWindows()

def scan_input(known_enc, known_names, input_path, input):

    # load the input image
    img_path = input_path + input
    input_img = face_recognition.load_image_file(img_path)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    # Find faces in the frame and get encodings
    face_locations = face_recognition.face_locations(input_img)
    shown_enc = face_recognition.face_encodings(input_img, face_locations)

    # make a named window
    cv2.namedWindow('Scanning Input Image', cv2.WINDOW_AUTOSIZE)

    # Continuous loop
    while cv2.getWindowProperty('Scanning Input Image', 0) >= 0:
        
        # compare all face encodings in img
        for (top, right, bottom, left), face_enc in zip(face_locations, shown_enc):
            matches = face_recognition.compare_faces(known_enc, face_enc)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_enc, face_enc)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_names[best_match_index]

            # Check if known or unknown
            if name is "Unknown":
                # Draw a rectangle around each face in img
                cv2.rectangle(img, (left, top), (right,bottom), (0,0, 255), 1)

                cv2.rectangle(img, (left, bottom - 35), (right, bottom), (0, 0, 255), -1)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(img, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
            else:
                # Draw a rectangle around each face in img
                cv2.rectangle(img, (left, top), (right,bottom), (0, 128, 0), 1)

                cv2.rectangle(img, (left, bottom - 35), (right, bottom), (0, 128, 0), -1)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(img, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Scanning Input Image', img)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # close all windows
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
    