# libraries imported
import os
import re
import face_recognition
from PIL import Image, ImageDraw

root_img_path = "assets/imgs/"
know_users_path = root_img_path + "users/"
known_encodings = []
known_names = []
known_files = []

unknown_path = root_img_path + "unknown-users/user-two.jpeg"
unknown_image = face_recognition.load_image_file(unknown_path)


def clean_files(files):
    for item in files:
        if item.startswith('.'):
            files.remove(item)


# function for importing known image files
def read_know_files():
    # add files of known users to a list
    for root_path, directory, file_names in os.walk(know_users_path):
        clean_files(file_names)
        known_files.extend(file_names)


# function for importing known names from files
def read_known_names():
    # add known users names to a list
    for file_name in known_files:
        face = face_recognition.load_image_file(know_users_path + file_name)
        known_names.append(file_name[:-4])
        known_encodings.append(face_recognition.face_encodings(face)[0])


def facial_detection(image):
    coordinates = face_recognition.face_locations(image)
    encodings = face_recognition.face_encodings(image)

    return coordinates, encodings

'''
def facial_recognition(drawing_instance, image, encodings, locations, draw):
    for (top, right, bottom, left), enc in zip(locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, enc)

        name = "Unknown"

        if True in matches:
            first = matches.index(True)
            name = known_names[first]

        draw.rectangle((left, top), (right, bottom), outline=(0,0,0))

        text_width, text_height = draw.textsize(name)
        draw.rectangle((left, bottom - text_height - 10), (right, bottom), fill=(0,0,0), outline=(0,0,0))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255,255,255,255))
'''

read_know_files()
read_known_names()

result_image = Image.fromarray(unknown_image)
face_coordinates, face_encodings = facial_detection(unknown_image)
writer = ImageDraw.Draw(result_image)

# facial_recognition(writer, result_image, known_encodings, face_coordinates, writer)

for (top, right, bottom, left), enc in zip(face_coordinates, face_encodings):
    matches = face_recognition.compare_faces(known_encodings, enc)

    name = "Unknown"

    if True in matches:
        first = matches.index(True)
        name = known_names[first]

    writer.rectangle(((left, top), (right, bottom)), outline=(255,255,0))

    text_width, text_height = writer.textsize(name)
    writer.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0,0,0), outline=(0,0,0))
    writer.text((left + 6, bottom - text_height - 5), name, fill=(255,255,255))


del writer
result_image.show()





