# libraries imported
import os
import re
import cv2
import face_recognition
from encoding import read_known_user, get_image_encodings
import numpy as np
import matplotlib.pyplot as plt

known_users_path = "appdata/imgs/users/"
input_path = "appdata/imgs/known-users/"
known_users = []
num_tests = 5
test_iter_correct = []
test_iter_incorrect = []

# train the faces
read_known_user(known_users)
known_enc = get_image_encodings(known_users_path)

for i in range(num_tests):

    # make counters for correct and incorrect recognitions
    correct = 0
    incorrect = 0

    # iterate through all users
    for i in len(known_users):

        # iterate through all of the images in test directory
        for img in os.listdir(input_path):

            # load the input image
            input_img = face_recognition.load_image_file(input_path + img)

            # find face in frame and get encodings
            face_locations, shown_enc = facial_detection(input_img)

            # match the faces to known faces
            info = match(shown_enc, known_enc, known_users)

            # get the name of the person in photo from file name
            name_in_file = img[:-5]

            # compare the name of current user and the name in the image
            if known_user[i][0] == name_in_file:
                # if the same then check if match gave the correct match

                if known_user[i][0] == info[0]:
                    # increment correct counter for correct match
                    correct++
                else:
                    # increment incorrect counter for not matching properly
                    incorrect++
            else:
                # if different then check if match gave unknown or different match

                if known_user[i][0] == info[0]:
                    # increment incorrect counter for incorrect match
                    incorrect++
                else:
                    # increment correct counter for not matching
                    correct++
    
    # add number of correct and incorrect matches to test_iter arrays
    test_iter_correct.append(correct)
    test_iter_incorrect.append(incorrect)

# create plot
fig, ax = plt.subplots()
index = np.arange(num_tests)
bar_width = 0.35
opacity = 0.8

# create correct and incorrect rectangles
rects1 = plt.bar(index, test_iter_correct, bar_width,
alpha=opacity,
color='g',
label='Correct matches')

rects2 = plt.bar(index + bar_width, test_iter_incorrect, bar_width,
alpha=opacity,
color='r',
label='Incorrect matches')

# labels
plt.xlabel('Test iteration')
plt.ylabel('Number of matches')
plt.title('Correct vs Incorrect matches of existing users')
plt.xticks(index + bar_width, ('1', '2', '3', '4', '5'))
plt.legend()

# make plot
plt.tight_layout()
plt.savefig('intended_matches.jpg', dpi=400)
plt.show()