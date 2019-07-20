
from matplotlib import pyplot
from scipy.misc import toimage
from keras.datasets import cifar10 
import cv2
import glob
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import keras
from keras.models import Sequential
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import cifar10
from keras import regularizers
from keras.callbacks import LearningRateScheduler
import numpy as np
from keras.models import model_from_json

PATH_DATA = "/home/driverless/ras_perception/DL_training/image_dataset_master_cropped_shape/"


def show_imgs(X):
    pyplot.figure(1)
    k = 0
    for i in range(0,6):
        for j in range(0,6):
            pyplot.subplot2grid((6,6),(i,j))
            pyplot.imshow(toimage(X[k]))
            k = k+1
    # show the plot
    pyplot.show()
 

def load_custom_data():
    x_list = []
    y_list = []

    # x_test_list = []
    # y_test_list = []

    ### LOAD Training DATA  ###
    print('##########    LOADING DATA     ##########')
    for dirname in os.listdir(PATH_DATA):
        if dirname == 'Ball':
            label = 0
        elif dirname == 'Cube':
            label = 1
        elif dirname == 'Cylinder':
            label = 2
        elif dirname == 'Hollow Cube':
            label = 3
        elif dirname == 'Cross':
            label = 4
        elif dirname == 'Triangle':
            label = 5
        elif dirname == 'Star':
            label = 6

        print('##########    ' + dirname + ' , Label : ' + str(label) + '      ##########')
        for file in glob.glob(PATH_DATA + dirname + "/*.jpg"):

            image = cv2.imread(file)

            x_list.append(cv2.resize(image, (32,32)))
            y_list.append(label)
        
    x_arr = np.array(x_list)
    y_arr = np.reshape(np.array(y_list), (-1,1))

    x_train, x_test, y_train, y_test = train_test_split(x_arr, y_arr, test_size=0.3, random_state=0)
    return (x_train, y_train), (x_test, y_test) 

(x_train, y_train), (x_test, y_test) = load_custom_data()
 
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
 
# mean-std normalization
mean = np.mean(x_train,axis=(0,1,2,3))
std = np.std(x_train,axis=(0,1,2,3))
x_train = (x_train-mean)/(std+1e-7)
x_test = (x_test-mean)/(std+1e-7)
 
 
# Load trained CNN model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights('model.h5')
 
#labels =  ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
 
labels = ['Ball', 'Cube', 'Cylinder', 'Hollow Cube', 'Cross', 'Triangle', 'Star' ]

indices = np.argmax(model.predict(x_test[:36]),1)
print( [labels[x] for x in indices])

show_imgs(x_test[:36])