import os
for dirname in os.listdir("/home/driverless/ras_perception/DL_training/image_dataset_keras_shape/Train/"):
    #if os.path.isdir(dirname):
#dirname = "/home/ajinkya/ras_ws/src/ras_perception/DL_training/image_dataset-master/Blue Hollow Cube"
    if dirname == 'Yellow Ball':
        count = 10001
    elif dirname == 'Yellow Cube':
        count = 20001
    elif dirname == 'Green Cube':
        count = 30001
    elif dirname == 'Green Cylinder':
        count = 40001
    elif dirname == 'Green Hollow Cube':
        count = 50001
    elif dirname == 'Orange Cross':
        count = 60001
    elif dirname == 'Orange Star':
        count = 70001
    elif dirname == 'Red Cylinder':
        count = 80001
    elif dirname == 'Red Hollow Cube':
        count = 90001
    elif dirname == 'Red Ball':
        count = 100001
    elif dirname == 'Blue Cube':
        count = 110001
    elif dirname == 'Blue Triangle':
        count = 120001
    elif dirname == 'Purple Cross':
        count = 130001
    elif dirname == 'Purple Star':
        count = 140001
       
    print('##########    ' + dirname + '    ##########')
    for i, filename in enumerate(os.listdir(dirname)):
        print('Name: ' + str(i + count))
        os.rename(dirname + "/" + filename, dirname + "/" + str(count + i) + ".jpg")

