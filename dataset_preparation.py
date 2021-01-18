############################################################################################
#
# Project:       Peter Moss Acute Myeloid & Lymphoblastic Leukemia AI Research Project
# Repository:    Tensorflow-ALL-Segmentation-2020
# Project:       Tensorflow-ALL-Segmentation-2020
#
# Author:        Aniruddh Sharma
# Title:         Dataset Preparation
# Description:   Dataset Preparation for the Tensorflow-ALL-Segmentation-2020
# License:       MIT License
# Last Modified: 2021-01-17
#
############################################################################################

import math
import cv2
import json,os,shutil
import numpy as np

color = (255,255,255)
thickness = -1
f_ext = '.png'
source = 'Class1_def'
folder = os.listdir(source)
num_of_images = len(folder)
if not os.path.isdir('processed_dataset'):
    new_folder = os.mkdir('processed_dataset')

for i in range(0,num_of_images):
    os.makedirs(os.path.join('processed_dataset', str(i+1)))

for j in range(0,num_of_images):
    os.makedirs(os.path.join('processed_dataset', str(j+1), 'mask'))

for k in range(0,num_of_images):
    os.makedirs(os.path.join('processed_dataset', str(k+1), 'image'))

defective_img = []
for l in range(0,num_of_images):
    defective_img.append(str(str(l+1)+'.png'))

for m in range(0,len(defective_img)):
    image_folder = 'processed_dataset/'+str(m+1)+'/'+'image/'
    shutil.copy(os.path.join(source,str(str(m+1)+'.png')), image_folder)


ellipse_details_file = 'labels.txt'
with open(ellipse_details_file) as fh:
    for line in fh:
        ellipse_details = line.split()

        axis_lengths = (int(float(ellipse_details[1])),int(float(ellipse_details[2])))
        angle = (180/math.pi)*float(ellipse_details[3])
        center = (int(float(ellipse_details[4])),int(float(ellipse_details[5])))

        img = np.zeros([512,512,1],dtype = np.uint8)
        new_img = cv2.ellipse(img, center, axis_lengths, angle, 0, 360, color, thickness)
        f_name = str(ellipse_details[0])
        new_name = '{}{}'.format(f_name, f_ext)
        filename = os.path.join('processed_dataset',str(ellipse_details[0]), 'mask', new_name)
        cv2.imwrite(filename, new_img)
