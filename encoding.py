# libraries imported
import os
import re
import face_recognition
import csv

known_users_path = "assets/imgs/users/"
known_encodings = {}
known_users = []


def read_known_user():
    with open('users.txt') as csv_file:
        file_reader = csv.reader(csv_file, delimiter=',')

        for line in file_reader:
            cur_user = []
            for i in range(len(line)):
                cur_user.append(line[i])
            known_users.append(cur_user)


def get_image_encodings():
    for file_name in os.listdir(known_users_path):
        face = face_recognition.load_image_file(known_users_path + file_name)
        index = int(file_name[:-4])
        known_encodings[index] = face_recognition.face_encodings(face)[0]

    print(known_encodings.keys())

'''
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
'''


read_known_user()
get_image_encodings()






