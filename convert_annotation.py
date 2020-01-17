# Script to convert the FDDB image dataset annotation file with elliptical face regions into rectangle
# bounding boxes in order to extract faces from images.


import os
import cv2
from math import *


# function to deal with if rectangle goes past the boundaries of the image
def boundary_cond(c, m):
    if c < 0:
        return 0
    elif c > m:
        return m
    else:
        return c


# merged all annotation and path files into one single file this line calls that merged file
input_annotation = os.path.join('FDDB-folds', 'allfolds_ellipse.txt')
output_annotation = os.path.join('FDDB-folds', 'allfolds_rectangle.txt')

# read lines from annotation file
with open(input_annotation) as annotation:
    data = [line.rstrip('\n') for line in annotation]

annotation = open(output_annotation, 'wb')
total_faces = 0
i = 0
# go line by line and import the data according to the readme provided by FDDB
while i < len(data):
    img_path = data[i]
    img_file = data[i] + '.jpg'
    img = cv2.imread(img_file)
    h = img.shape[0]
    w = img.shape[1]
    num_faces = int(data[i + 1])
    # count running total of number of faces in images
    total_faces = total_faces + num_faces
    # for each face in the image, import the annotations
    for j in range(num_faces):
        face = data[i + 2 + j].split()[0:5]
        r_major = float(face[0])
        r_minor = float(face[1])
        angle = float(face[2])
        c_x = float(face[3])
        c_y = float(face[4])
        # convert elliptical into rectangular coordinates
        tan_t = -(r_minor / r_major) * tan(angle)
        t = atan(tan_t)
        x1 = c_x + (r_major * cos(t) * cos(angle) - r_minor * sin(t) * sin(angle))
        x2 = c_x + (r_major * cos(t + pi) * cos(angle) - r_minor * sin(t + pi) * sin(angle))
        x_max = boundary_cond(max(x1, x2), w)
        x_min = boundary_cond(min(x1, x2), w)
        if tan(angle) != 0:
            tan_t = (r_minor / r_major) * (1 / tan(angle))
        else:
            tan_t = (r_minor / r_major) * (1 / (tan(angle) + 0.0001))
        t = atan(tan_t)
        y1 = c_y + (r_minor * sin(t) * cos(angle) + r_major * cos(t) * sin(angle))
        y2 = c_y + (r_minor * sin(t + pi) * cos(angle) + r_major * cos(t + pi) * sin(angle))
        y_max = boundary_cond(max(y1, y2), h)
        y_min = boundary_cond(min(y1, y2), h)
        # write rectangular annotations to file
        text = img_file + ',' + str(x_min) + ',' + str(y_min) + ',' + str(x_max) + ',' + str(y_max) + '\n'
        text_b = text.encode()
        annotation.write(text_b)

    i = i + num_faces + 2

annotation.close()
