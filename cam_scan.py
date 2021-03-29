# libraries imported
import os
import re
import cv2
import numpy as np
import face_recognition
from encoding import read_known_user, get_image_encodings


def main():
    known_users_path = "appdata/imgs/users/"
    input_path = "appdata/imgs/inputs/"
    # known_enc = []
    known_users = []
    # known_files = []

    # test image
    # inp = "user-one.jpg"

    # train the faces
    read_known_user(known_users)
    known_enc = get_image_encodings(known_users_path)

    # Ask user if they want a live scan or scan from an image
    choice = input("Would you like to scan live (type \"L\") or from an image (type \"I\")?: ")

    if choice.lower() == "L".lower():
        # live scan mode
        live_scan(known_enc, known_users)
    elif choice.lower() == "I".lower():
        # get the name of the file
        inp = input("Enter the file name: ")

        # input mode
        scan_input(known_enc, known_users, input_path, inp)
    else:
        print("Invalid option")

    quit()


def live_scan(known_enc, known_users):

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
        face_locations, shown_enc = facial_detection(rgb_frame)

        # match the face to known faces
        info = match(shown_enc, known_enc, known_users)

        # display the information
        display(face_locations, info[0], info[1], info[2], info[3], frame, 1)

        # Display the resulting image
        cv2.imshow('Live Scan', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # end capture and close windows
    vid_cap.release()
    cv2.destroyAllWindows()


def scan_input(known_enc, known_users, input_path, input):

    # load the input image
    img_path = input_path + input
    input_img = face_recognition.load_image_file(img_path)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    # Find faces in the frame and get encodings
    face_locations, shown_enc = facial_detection(input_img)

    # make a named window
    cv2.namedWindow('Scanning Input Image', cv2.WINDOW_AUTOSIZE)

    # Continuous loop
    while cv2.getWindowProperty('Scanning Input Image', 0) >= 0:
        
        # match the face to known faces
        info = match(shown_enc, known_enc, known_users)

        # display the information
        display(face_locations, info[0], info[1], info[2], info[3], img, 1)

        # Display the resulting image
        cv2.imshow('Scanning Input Image', img)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # close all windows
    cv2.destroyAllWindows()


def facial_detection(image):
    coordinates = face_recognition.face_locations(image)
    encodings = face_recognition.face_encodings(image, coordinates)

    return coordinates, encodings

def match(shown_enc, known_enc, known_users):

    name = "Unknown"
    sex = "Unknown"
    occ = "Unknown"
    bday = "Unknown"
    num_faces = len(shown_enc)

    # check if list is empty
    if len(shown_enc) != 0:
        # compare all face encodings in img
        for face_enc in shown_enc:
            matches = face_recognition.compare_faces(known_enc, face_enc)
            name = "Unknown"
            sex = "Unknown"
            occ = "Unknown"
            bday = "Unknown"

            face_distances = face_recognition.face_distance(known_enc, face_enc)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_users[best_match_index][0]
                sex = known_users[best_match_index][1]
                occ = known_users[best_match_index][2]
                bday = known_users[best_match_index][3]

            return [name, sex, occ, bday, num_faces]

    else:
        return [name, sex, occ, bday, num_faces]

def display(face_locations, name, sex, occ, bday, img, ratio):
    
    # check if list is empty
    if len(face_locations) != 0:
        # compare all face encodings in img
        for (top, right, bottom, left) in face_locations:
            
            left = int(left * ratio)
            right = int(right * ratio)
            top = int(top * ratio)
            bottom = int(bottom * ratio)

            # Check if known or unknown
            if name == "Unknown":
                # Draw a rectangle around each face in img
                cv2.rectangle(img, (left, top), (right,bottom), (0,0, 255), 1)

                cv2.rectangle(img, (left, bottom + 40), (right, bottom), (0, 0, 255), -1)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(img, name, (left + 6, bottom + 17), font, 0.5, (255, 255, 255), 1)
            else:
                # Draw a rectangle around each face in img
                cv2.rectangle(img, (left, top), (right,bottom), (0, 128, 0), 1)

                cv2.rectangle(img, (left, bottom + 80), (right, bottom), (0, 128, 0), -1)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(img, name, (left + 6, bottom + 17), font, 0.5, (255, 255, 255), 1)
                cv2.putText(img, sex, (left + 6, bottom + 34), font, 0.5, (255, 255, 255), 1)
                cv2.putText(img, occ, (left + 6, bottom + 51), font, 0.5, (255, 255, 255), 1)
                cv2.putText(img, bday, (left + 6, bottom + 68), font, 0.5, (255, 255, 255), 1)


if __name__ == '__main__':
    main()