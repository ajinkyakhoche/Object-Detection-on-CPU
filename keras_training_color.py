'''Train a simple deep CNN on the CIFAR10 small images dataset.
It gets to 75% validation accuracy in 25 epochs, and 79% after 50 epochs.
(it's still underfitting at that point, though).
'''

from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
import glob
import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

'''
@author:Ajinkya Khoche

# of classes Based on "Object dataset" in Modules
0:  Yellow
1:  Green
2:  Orange
3:  Red
4:  Blue
5:  Purple
'''

batch_size = 32
num_classes = 7
epochs = 100
data_augmentation = True
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_RAS_model_color_3.h5'

PATH_DATA = "../image_dataset_keras_color/"

color_class = ['Yellow', 'Green', 'Orange', 'Red', 'Blue', 'Purple', 'Nothing']


def load_custom_data():
    x_list = []
    y_list = []

    # x_test_list = []
    # y_test_list = []

    ### LOAD TEST DATA  ###
    print('##########    LOADING DATA     ##########')
    for dirname in os.listdir(PATH_DATA):
        if dirname == 'Yellow':
            label = 0
        elif dirname == 'Green':
            label = 1
        elif dirname == 'Orange':
            label = 2
        elif dirname == 'Red':
            label = 3
        elif dirname == 'Blue':
            label = 4
        elif dirname == 'Purple':
            label = 5
        elif dirname == 'Nothing':
            label = 6

        print('##########    ' + dirname + ' , Label : ' + str(label) + '      ##########')
        for file in glob.glob(PATH_DATA + dirname + "/*.jpg"):

            image = cv2.imread(file)

            x_list.append(cv2.resize(image, (32,32)))
            y_list.append(label)
        
    x_arr = np.array(x_list)
    y_arr = np.reshape(np.array(y_list), (-1,1))

    x_train, x_test, y_train, y_test = train_test_split(x_arr, y_arr, test_size=0.4, random_state=0)
    return (x_train, y_train), (x_test, y_test) 


# The data, split between train and test sets:
#(x_train, y_train), (x_test, y_test) = cifar10.load_data()
(x_train, y_train), (x_test, y_test) = load_custom_data()
#load_custom_data()

# Shuffle data to randomize
# (x_train, y_train) = shuffle(x_tr, y_tr, random_state=0)
# (x_test, y_test) = shuffle(x_te, y_te, random_state=0)

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# model.add(Conv2D(64, (3, 3), padding='same'))
# model.add(Activation('relu'))
# model.add(Conv2D(64, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        shear_range=0.,  # set range for random shear
        zoom_range=0.,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        workers=4)

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

for file in glob.glob('../RAS_DATASET'+ "/*.jpg"):
    image = cv2.imread(file)

    input_img = []
    input_img.append(cv2.resize(image, (32,32)))
    input_img = np.array(input_img)

    prediction = model.predict(input_img)
    #print('Actual: ' + str(dirname) + '     detected: ' + shape_class[np.argmax(prediction)])
    print('detected: ' + color_class[np.argmax(prediction)])
    cv2.imshow('image', cv2.resize(image, (640,480)))
    cv2.waitKey(3000)
