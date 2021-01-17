############################################################################################
#
# Project:       Peter Moss Acute Myeloid & Lymphoblastic Leukemia AI Research Project
# Repository:    Tensorflow-ALL-Segmentation-2020
# Project:       Tensorflow-ALL-Segmentation-2020
#
# Author:        Aniruddh Sharma
# Title:         Model Class
# Description:   Model functions for the Tensorflow-ALL-Segmentation-2020
# License:       MIT License
# Last Modified: 2021-01-17
#
############################################################################################

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Lambda, Conv2DTranspose, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import backend as K
from scipy.ndimage.measurements import label
import numpy as np
from skimage.io import imshow
import random,os,cv2,sklearn,time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split

from Classes.Helpers import Helpers

class Model():
    """ Model Class
    Model functions for the Tensorflow-ALL-Segmentation-2020.
    """

    def __init__(self):
        """ Initializes the class. """

        self.Helpers = Helpers("Model", False)

    def do_model(self):

        os.environ["CUDA_VISIBLE_DEVICES"]="-1"

        # gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction = 0.9)
        # sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

        # PATH = os.path.join('logs', 'image_segmentation')

        # tensorboard = TensorBoard(log_dir=PATH, profile_batch = 0)

        img_height = 512
        img_width = 512
        img_channel = 3

        img_num = len(os.listdir('dataset/images'))

        images = np.empty([img_num,img_height,img_width,img_channel])
        masks = np.empty([img_num,img_height,img_width,1])
        k=0
        for i in os.listdir('dataset/images'):
            img_file = cv2.resize(mpimg.imread(os.path.join('dataset/images',i)),(img_height,img_width))*255
            images[k] = np.reshape(img_file,(img_height,img_width,img_channel))
            k+=1
        m=0
        for i in os.listdir('dataset/mask'):
            f_name, f_ext = os.path.splitext(i)
            is_1 = f_name[-1]
            img_file = cv2.resize(mpimg.imread(os.path.join('dataset/mask',i)),(img_height,img_width))//250
            masks[m] = np.reshape(img_file,(img_height,img_width,1))
            masks[m,0,0,0] = 0
            if int(is_1)==1:
                masks[m,0,0,0] = 1
            m+=1

        X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size=0.1)

        # print(X_train.shape, X_test.shape)
        model = Sequential()
        input_tensor = Input((512, 512, 3))
        conv_layer_1 = Conv2D(16, (3, 3), activation='relu', padding='same')(input_tensor)
        conv_layer_1 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv_layer_1)
        pool_layer_1 = MaxPooling2D(pool_size=(2, 2))(conv_layer_1)

        conv_layer_2 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool_layer_1)
        conv_layer_2 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv_layer_2)
        pool_layer_2 = MaxPooling2D(pool_size=(2, 2))(conv_layer_2)

        conv_layer_3 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool_layer_2)
        conv_layer_3 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv_layer_3)

        up_layer_1 = concatenate([Conv2DTranspose(16, kernel_size=(
            2, 2), strides=(2, 2), padding='same')(conv_layer_3), conv_layer_2], axis=3)
        conv_layer_4 = Conv2D(32, (3, 3), activation='relu', padding='same')(up_layer_1)
        conv_layer_4 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv_layer_4)

        up_layer_2 = concatenate([Conv2DTranspose(8, kernel_size=(
            2, 2), strides=(2, 2), padding='same')(conv_layer_4), conv_layer_1], axis=3)
        conv_layer_5 = Conv2D(16, (3, 3), activation='relu', padding='same')(up_layer_2)
        conv_layer_5 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv_layer_5)

        conv_layer_6 = Conv2D(1, (1, 1), activation='sigmoid')(conv_layer_5)

        model = Model(inputs=input_tensor, outputs=conv_layer_6)

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                                                                             tf.keras.metrics.AUC(name='auc')])
        model.fit(X_train, y_train, batch_size=5, epochs=5, verbose=1, validation_split=0.1)
        self.Helpers.logger.info("Model training complete.")



    def get_prediction(self, model, test_img_folder):
        """ Gets a prediction for an image. """
        img_height = 512
        img_width = 512
        img_channel = 3

        img_num = len(os.listdir(test_img_folder))
        images = np.empty([img_num,img_height,img_width,img_channel])
        k=0
        for i in os.listdir(test_img_folder):
            img_file = cv2.resize(mpimg.imread(os.path.join('dataset/images',i)),(img_height,img_width))*255
            images[k] = np.reshape(img_file,(img_height,img_width,img_channel))
            k+=1
        model = tf.keras.models.load_model(model)
        predict = model.predict(images)
        img_pred = np.array(255*predict, dtype = np.uint8)
        labels = img_pred[:,0,0,0]

        return img_pred, labels
