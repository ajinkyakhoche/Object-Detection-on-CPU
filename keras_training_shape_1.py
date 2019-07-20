from matplotlib import pyplot
# from scipy.misc import toimage
from keras.datasets import cifar10 
import cv2
import glob
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def show_imgs(X):
    pyplot.figure(1)
    k = 0
    for i in range(0,6):
        for j in range(0,6):
            pyplot.subplot2grid((6,6),(i,j))
            # pyplot.imshow(toimage(X[k]))
            pyplot.imshow(X[k])
            k = k+1
    # show the plot
    pyplot.show()
 
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
# show_imgs(x_test[:16])

shape_class = ['Ball', 'Cube', 'Cylinder', 'Hollow Cube', 'Cross', 'Triangle', 'Star' ]

PATH_DATA = "/media/ajinkya/My Passport/RAS/ras_perception/DL_training/image_dataset_keras_shape"

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
        for file in glob.glob(os.path.join(PATH_DATA, dirname, "*.jpg")):

            image = cv2.imread(file)

            x_list.append(cv2.resize(image, (32,32)))
            y_list.append(label)
        
    x_arr = np.array(x_list)
    y_arr = np.reshape(np.array(y_list), (-1,1))

    x_train, x_test, y_train, y_test = train_test_split(x_arr, y_arr, test_size=0.3, random_state=0)
    return (x_train, y_train), (x_test, y_test) 




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
 
def lr_schedule(epoch):
    lrate = 0.001
    if epoch > 75:
        lrate = 0.0005
    elif epoch > 100:
        lrate = 0.0003       
    return lrate
 
(x_train, y_train), (x_test, y_test) = load_custom_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')


from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator( rotation_range=90, 
                 width_shift_range=0.1, height_shift_range=0.1, 
                 horizontal_flip=True) 
datagen.fit(x_train)

#z-score
mean = np.mean(x_train,axis=(0,1,2,3))
std = np.std(x_train,axis=(0,1,2,3))
x_train = (x_train-mean)/(std+1e-7)
x_test = (x_test-mean)/(std+1e-7)
 
num_classes = 7
y_train = np_utils.to_categorical(y_train,num_classes)
y_test = np_utils.to_categorical(y_test,num_classes)
 
weight_decay = 1e-4
model = Sequential()
model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=x_train.shape[1:]))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
 
model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))
 
model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))
 
model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))
 
model.summary()
 
#data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    )
datagen.fit(x_train)
 
#training
batch_size = 64
 
opt_rms = keras.optimizers.rmsprop(lr=0.001,decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=opt_rms, metrics=['accuracy'])
model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),\
                    steps_per_epoch=x_train.shape[0] // batch_size,epochs=125,\
                    verbose=1,validation_data=(x_test,y_test),callbacks=[LearningRateScheduler(lr_schedule)])
#save to disk
model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)
model.save_weights('model.h5') 
  
#testing
scores = model.evaluate(x_test, y_test, batch_size=128, verbose=1)
print('\nTest result: %.3f loss: %.3f' % (scores[1]*100,scores[0]))

### Test


indices = np.argmax(model.predict(x_test[:36]),1)
print( [shape_class[kk] for kk in indices])
show_imgs(x_test[:36])