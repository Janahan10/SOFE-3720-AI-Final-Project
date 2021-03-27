# libraries imported
import os
import re
import cv2
import numpy as np
import face_recognition
from encoding import read_known_user, get_image_encodings

known_users_path = "../appdata/imgs/users/"
input_path = "../appdata/imgs/inputs/"
# known_enc = []
known_users = []

# train the faces
read_known_user(known_users)
known_enc = get_image_encodings(known_users_path)

# make counters for correct and incorrect recognitions
correct = 0
incorrect = 0

# # iterate through all users
# for i in len(known_users):

# iterate through all of the images in test directory
for img in os.listdir(input_path):

    print(img)
