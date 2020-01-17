# Script for extracting face and non-face image patches from the image files using the FDDB annotations

import os
import cv2
import numpy as np


# check if image directory is created, create it if not
def check_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


# function to find the boundaries of non-face image patches
def non_boundary(im_min, im_max, crop, dim):
    if im_min > im_crop_size[1]:
        non_max = im_min - 1
        non_min = non_max - crop
    elif w - im_max > crop:
        non_max = dim
        non_min = dim - crop
    else:
        non_min = 0
        non_max = 0
    return non_min, non_max


check_dir(os.path.join(os.getcwd(), 'Extracted_Faces'))
check_dir(os.path.join(os.getcwd(), 'Extracted_Non'))
annotation_file = os.path.join('FDDB-folds', 'allfolds_rectangle.txt')
im_crop_size = (60, 60)

with open(annotation_file, 'r') as file:
    idx = 0
    for line in file:
        annotations = line.split(',')
        img_file = annotations[0]
        x_min = int(float(annotations[1]))
        y_min = int(float(annotations[2]))
        x_max = int(float(annotations[3]))
        y_max = int(float(annotations[4].rstrip('\n')))
        img = cv2.imread(img_file)
        h = img.shape[0]
        w = img.shape[1]
        # extract portion of image within bounding box annotation (with face)
        img_face = img[y_min:y_max, x_min:x_max]
        # check for empty image
        if img_face.size == 0:
            continue
        # resize image
        img_face_scale = cv2.resize(img_face, im_crop_size, interpolation=cv2.INTER_AREA)
        img_out_file = 'face' + str(idx) + '.jpg'
        img_path_face = os.path.join('Extracted_Faces', img_out_file)
        cv2.imwrite(img_path_face, img_face_scale)

        # extract non-face images
        # find boundaries for non-face images (background outside of bounding box)
        [xn_min, xn_max] = non_boundary(x_min,x_max,im_crop_size[0], w)
        if xn_min == 0 and xn_max == 0:
            continue
        [yn_min, yn_max] = non_boundary(y_min, y_max, im_crop_size[1], h)
        if yn_min == 0 and yn_max == 0:
            continue
        img_non = img[yn_min:yn_max, xn_min:xn_max]
        img_out_file = 'non' + str(idx) + '.jpg'
        img_path_non = os.path.join('Extracted_Non', img_out_file)
        cv2.imwrite(img_path_non, img_non)
        idx += 1



