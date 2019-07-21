import cv2
import numpy as np 
import keras.models
from keras.models import model_from_json 
import glob
from datetime import datetime

'''
@author:Ajinkya Khoche

model_shape: A CNN which detects shape
-----------
Class Labels:   SHAPE
                -----
0:  Ball
1:  Cube
2:  Cylinder
3:  Hollow Cube
4:  Cross
5:  Triangle
6:  Star
########################################

model_color: A CNN which detects color
-----------
Class Labels:   COLOR
                ------
0:  Yellow
1:  Green
2:  Orange
3:  Red
4:  Blue
5:  Purple
'''
# Load SHAPE CNN
# model_shape = keras.models.load_model('./saved_models/keras_RAS_model_shape_1.h5')
json_file = open('./saved_models/cropped_shape_3.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model_shape = model_from_json(loaded_model_json)
model_shape.load_weights('./saved_models/cropped_shape_3.h5')

# Load COLOR CNN
model_color = keras.models.load_model('./saved_models/keras_cropped_color_2.h5')

shape_class = ['Ball', 'Cube', 'Cylinder', 'Hollow Cube', 'Cross', 'Triangle', 'Star' ]
color_class = ['Yellow', 'Green', 'Orange', 'Red', 'Blue', 'Purple']

VIDEO_INFERENCE = 1
IMG_INFERNECE = 0

N_SHAPES = 7
N_COLORS = 6

DEBUG = 1

def morphOpen(image):
    # define structuring element
    # take 5% of least dimension of image as kernel size
    kernel_size = min(5, int(min(image.shape[0],image.shape[1])*0.05))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel_size,kernel_size))
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return opening

def detect_object(image, color_label):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    if color_label == 0:    #YELLOW
        mask = cv2.inRange(hsv, np.array([17,128,0]), np.array([32,255,255]))
    elif color_label == 1:  #GREEN
        mask = cv2.inRange(hsv, np.array([32,129,0]), np.array([66,255,255]))
    elif color_label == 2:  #ORANGE
        # mask1 = cv2.inRange(hsv, np.array([0,40,120]), np.array([8,255,255]))
        # mask2 = cv2.inRange(hsv, np.array([160,40,120]), np.array([179,255,255]))
        # mask = mask1 + mask2 
        mask = cv2.inRange(hsv, np.array([0,175,188]), np.array([23,255,255]))
    elif color_label == 3:  #RED
        # mask1 = cv2.inRange(hsv, np.array([0,50,50]), np.array([15,255,255]))
        # mask2 = cv2.inRange(hsv, np.array([170,50,50]), np.array([180,255,255]))
        # mask = mask1 + mask2 
        mask = cv2.inRange(hsv, np.array([0,150,0]), np.array([4,255,200]))
    elif color_label == 4:  #BLUE
        mask = cv2.inRange(hsv, np.array([68,0,0]), np.array([110,255,255]))
    elif color_label == 5:  #PURPLE
        mask = cv2.inRange(hsv, np.array([116,100,8]), np.array([179,166,173]))

    if DEBUG:
        cv2.imshow('mask', mask)
        cv2.waitKey(0)

    # blur image
    #mask_blur = cv2.GaussianBlur(mask,(2,2),0)
    mask_morph = morphOpen(mask)

    if DEBUG: 
        cv2.imshow('mask_blur_morph', mask_morph)
        cv2.waitKey(0)

    # find contours
    # _, contours, _ = cv2.findContours(mask_morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours, _ = cv2.findContours(mask_morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for k in range(contours.__len__()):
        xx, yy, w, h = cv2.boundingRect(contours[k])
        #print(box[2]*box[3])

        if w/h > 0.5 and w/h<2:
            if w*h > 2000:
                # we need to give slightly bigger image to detector to get a clear detection
                tl_x = max(0, int(xx - 0.25*w))
                tl_y = max(0, int(yy - 0.25*h))
                br_x = min(image.shape[1], int(xx + w + 0.25*w))
                br_y = min(image.shape[0], int(yy + h + 0.25*h))

                #bBox_img = image[yy:yy+h, xx:xx+w]
                bBox_img = image[tl_y:tl_y+(br_y-tl_y), tl_x:tl_x+(br_x-tl_x)]
                input_img = []
                input_img.append(cv2.resize(bBox_img, (32,32)))
                input_img = np.array(input_img)

                pred_shape = model_shape.predict(input_img)
                pred_color = model_color.predict(input_img)

                draw_result([xx,yy,w,h], image, pred_shape, pred_color)
                # draw_result([xx,yy,w,h], image)
                # draw_result([tl_x,tl_y,tl_x+(br_x-tl_x),tl_y+(br_y-tl_y)], image)
                
                print(color_class[np.argmax(pred_color)]+ ' ' +shape_class[np.argmax(pred_shape)])
    
def draw_result(box, image, pred_shape, pred_color):
# def draw_result(box, image):
    cv2.rectangle(image, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0,255,0), 2)
    '''Both shape and color on bounding box'''
    cv2.putText(image, color_class[np.argmax(pred_color)]+ ' ' +shape_class[np.argmax(pred_shape)], (box[0],box[1]-15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 2,
cv2.LINE_AA)
    '''only color on bounding box'''
#     cv2.putText(image, color_class[np.argmax(pred_color)], (box[0],box[1]-15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 2,
# cv2.LINE_AA)
    '''Only shape on bounding box'''
#     cv2.putText(image, shape_class[np.argmax(pred_shape)], (box[0],box[1]-15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 2,
# cv2.LINE_AA)


if VIDEO_INFERENCE:    
    cap = cv2.VideoCapture('../ras_labeling/vid2.mp4')
    #cap = cv2.VideoCapture(0)

    while cap.isOpened():
        a = datetime.now()

        ret, image = cap.read()
        
        image = cv2.resize(image, (0,0), fx=0.33, fy=0.33)

        for i in range(N_COLORS):
            detect_object(image, i)
        
        cv2.imshow('result', image)
        cv2.waitKey(1)
        
        b = datetime.now()
        c = b - a
        fps = 1.0/(c.total_seconds())
        print('## FPS: ' + str(fps))
        print('')

elif IMG_INFERNECE:
    try:
        while True:
            for file in glob.glob('../RAS_DATASET' + "/*.jpg"):
                image = cv2.imread(file)
                for i in range(N_COLORS):
                    detect_object(image, i)
                
                cv2.imshow('result', image)
                cv2.waitKey(0)            

    except KeyboardInterrupt:
        pass

    # image = cv2.imread('./1001.jpg')
    # image = cv2.resize(image, (0,0), fx=0.33, fy=0.33)

    # for i in range(N_COLORS):
    #     detect_object(image, i)
    
    # cv2.imshow('result', image)
    # cv2.waitKey(0)



