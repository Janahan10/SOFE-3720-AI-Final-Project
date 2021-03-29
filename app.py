# libraries imported
import os
import re
import cv2
import numpy as np
import imutils
import face_recognition
from encoding import read_known_user, get_image_encodings
from cam_scan import facial_detection, match, display
import PySimpleGUI as sg

def main():
    known_users_path = "appdata/imgs/users/"
    known_users = []

    # train the faces
    print("Training and encoding faces...")
    read_known_user(known_users)
    known_enc = get_image_encodings(known_users_path)
    print("Finished encoding")

    sg.theme("DarkBlue")

    file_list_column = [
        [
            sg.Text("Image Folder"),
            sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
            sg.FolderBrowse(),
            sg.Checkbox("Live Scan", size=(10, 1), key="-LIVE-")
        ],
        [
            sg.Listbox(
                values=[], enable_events=True, size=(40, 20), key="-FILE LIST-"
            )
        ],
        [sg.Button("Exit", size=(10, 1))]
    ]

    # For now will only show the name of the file that was chosen
    image_viewer_column = [
        [sg.Text("Choose an image from list on left or do a live scan:")],
        [sg.Text(size=(40, 1), key="-TOUT-")],
        [sg.Image(key="-IMAGE-")],
    ]

    # ----- Full layout -----
    layout = [
        [
            sg.Column(file_list_column),
            sg.VSeperator(),
            sg.Column(image_viewer_column),
        ]
    ]

    # Create the window and show it without the plot
    window = sg.Window("Facial Recognition System", layout, location=(0, 0), resizable=True)

    cap = cv2.VideoCapture(0)

    while True:
        event, values = window.read(timeout=20)
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        
        if values["-LIVE-"]:
            event = "None"

            imgbytes = live_scan(cap, known_enc, known_users)
            window["-IMAGE-"].update(data=imgbytes)
        elif event == "-FILE LIST-":  # A file was chosen from the listbox

            try:
                img_path = os.path.join(
                    values["-FOLDER-"], values["-FILE LIST-"][0]
                )

            except:
                pass

            imgbytes = img_scan(img_path, known_enc, known_users)
            window["-IMAGE-"].update(data=imgbytes)

        if event == "-FOLDER-":
            window.FindElement("-LIVE-").Update(value=False)
            folder = values["-FOLDER-"]
            try:
                # Get list of files in folder
                print(folder)
                file_list = os.listdir(folder)
                
            except:
                print(folder)
                file_list = []

            fnames = [
                f
                for f in file_list
                if os.path.isfile(os.path.join(folder, f))
                and f.lower().endswith((".png", ".gif", ".jpg"))
            ]
            window["-FILE LIST-"].update(fnames)

    window.close()

def img_scan(img_path, known_enc, known_users):
    # load the input image
    print("Loading image...")
    input_img = face_recognition.load_image_file(img_path)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    print("Image loaded")

    # Find faces in the frame and get encodings
    print("Processing image...")
    face_locations, shown_enc = facial_detection(input_img)

    # match the face to known faces
    print("Comparing faces...")
    info = match(shown_enc, known_enc, known_users)

    print("Comparing done")

    # display the information
    max_size = 500
    height = img.shape[0]
    width = img.shape[1]
    ratio = min(max_size/width, max_size/height)
    if height >= width:
        resized_img = imutils.resize(img, height=max_size)
        img = resized_img
    else:
        resized_img = imutils.resize(img, width=max_size)
        img = resized_img
    display(face_locations, info[0], info[1], info[2], info[3], img, ratio)

    imgbytes = cv2.imencode(".png", img)[1].tobytes()

    return imgbytes

def live_scan(cap, known_enc, known_users):

    print("Live processing...")

    # read the video capture
    _, frame = cap.read()

    # resize the frame to be proc and convert to rgb
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Find faces in the frame and get encodings
    face_locations, shown_enc = facial_detection(rgb_frame)

    # match the face to known faces
    info = match(shown_enc, known_enc, known_users)

    # display the information
    display(face_locations, info[0], info[1], info[2], info[3], frame, 1)

    imgbytes = cv2.imencode(".png", frame)[1].tobytes()
    return imgbytes

if __name__ == '__main__':
    main()
    