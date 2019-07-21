# Object-Detection-on-CPU
This work covers the perception pipeline of our submission for the Robotics and Autonomous Systems Project course [DD2425](https://www.kth.se/student/kurser/kurs/DD2425?l=en). The course involved constructing a mobile robot from scratch to carry out mock search and rescue. The goal was to detect and localize the objects in a scene on an intel NUC platform, **without a GPU**.  The objects were multi-colored toy blocks shown below: 

<p align="center">
<img src="./docs/ras_objects.jpg" width="500" height="375">
</p>

There are two ways to go about it. One can either use traditional image analysis (like thresholding, morphing, contour detection etc.). Although the approach sounds good on paper, it will never generalize well and be free of outliers. The thresholding parameters may change with the lighting. The shape detection (for instance between a *cube* and a *hollow cube*) and is non-trivial. Another approach is training a neural network. We started by looking at the excellent fork of yolov2 by [AlexeyAB](https://github.com/AlexeyAB/yolo2_light). We observed that even the lightest object detector network gave a measely 0.2 FPS with poor detection accuracy. 

## Approach
The key is to breakdown the problem of object detection into object localization and image classification (which many deep learning methods like R-CNN do). Using traditional image analysis, the algorithm can generate many regions of interest, which can be fed to an image recognition network for accurate detection. Moreover, we used separate CNN for color and shape detection for increased robustness. The pipeline is shown below:
