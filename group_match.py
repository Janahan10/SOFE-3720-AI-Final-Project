# libraries imported
import os
import re
import cv2
import face_recognition
from encoding import read_known_user, get_image_encodings
import numpy as np
import matplotlib.pyplot as plt

known_users_path = "appdata/imgs/users/"
input_path = "appdata/imgs/grouped/"
known_users = []
num_tests = 5
test_iter_all = []
test_iter_not_all = []
num_faces = 36

# train the faces
read_known_user(known_users)
known_enc = get_image_encodings(known_users_path)

for i in range(num_tests):

    # make counters for when all faces are matched and not
    all_m = 0
    not_all_m = 0

    # iterate through all of the images in test directory
    for img in os.listdir(input_path):

        # load the input image
        input_img = face_recognition.load_image_file(input_path + img)

        # find face in frame and get encodings
        face_locations, shown_enc = facial_detection(input_img)

        # match the faces to known faces
        info = match(shown_enc, known_enc, known_users)

        # check if number of faces matched is equal to intended number
        if num_faces == info[4]:
            # increment all_m counter
            all_m += 1
        else:
            # increment not_all_m counter
            not_all_m += 1
    
    # add number of all and not all matched to test_iter arrays
    test_iter_all.append(all_m)
    test_iter_not_all.append(not_all_m)

# create plot
fig, ax = plt.subplots()
index = np.arange(num_tests)
bar_width = 0.35
opacity = 0.8

# create all and not all matched rectangles
rects1 = plt.bar(index, test_iter_all, bar_width,
alpha=opacity,
color='g',
label='All faces matched')

rects2 = plt.bar(index + bar_width, test_iter_not_all, bar_width,
alpha=opacity,
color='r',
label='Not all faces matched')

# labels
plt.xlabel('Test iteration')
plt.ylabel('Number of complete matches')
plt.title('Successfulness of group scanning')
plt.xticks(index + bar_width, ('1', '2', '3', '4', '5'))
plt.legend()

# make plot
plt.tight_layout()
plt.savefig('group_match.jpg', dpi=400)
plt.show()