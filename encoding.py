# libraries imported
import os
import re
import face_recognition

know_users_path = "assets/imgs/users/"
known_encodings = []
known_names = []
known_files = []


def read_known_files(users_path, files):
    # add files of known users to a list
    for root_path, directory, file_names in os.walk(users_path):
        files.extend(file_names)


def read_known_names(users_path, files, names, encodings):
    # add known users names to a list
    for file_name in files:
        face = face_recognition.load_image_file(users_path + file_name)
        names.append(file_name[:-4])
        encodings.append(face_recognition.face_encodings(face)[0])


read_known_files(know_users_path, known_files)
read_known_names(know_users_path, known_files, known_names, known_encodings)

# print(known_names)



