# libraries imported
import os
import re
import face_recognition

know_users_path = "assets/imgs/users/"
known_encodings = []
known_names = []
known_files = []


def read_know_files():
    # add files of known users to a list
    for root_path, directory, file_names in os.walk(know_users_path):
        known_files.extend(file_names)


def read_known_names():
    # add known users names to a list
    for file_name in known_files:
        face = face_recognition.load_image_file(know_users_path + file_name)
        known_names.append(file_name[:-4])
        known_encodings.append(face_recognition.face_encodings(face)[0])


read_know_files()
read_known_names()

# print(known_names)



