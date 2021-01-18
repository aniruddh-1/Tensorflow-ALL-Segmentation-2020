############################################################################################
#
# Project:       Peter Moss Acute Myeloid & Lymphoblastic Leukemia AI Research Project
# Repository:    Tensorflow-ALL-Segmentation-2020
# Project:       Tensorflow-ALL-Segmentation-2020
#
# Author:        Aniruddh Sharma
# Title:         Mask Preparation
# Description:   Mask Image for defected cells for the Tensorflow-ALL-Segmentation-2020
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
source = 'dataset/defective'

if not os.path.isdir('test'):
    k = os.mkdir('test')

for i in range(135,150):
    os.makedirs(os.path.join('test', str(i+1)))

for i in range(135,150):
    os.makedirs(os.path.join('test', str(i+1), 'image'))

defective_img = []
for i in range(135,150):
    defective_img.append(str(str(i+1)+'.png'))
i=136
for img in defective_img:
    j = 'test/'+str(i)+'/'+'image/'
    shutil.copy(os.path.join(source,img), j)
    i+=1
