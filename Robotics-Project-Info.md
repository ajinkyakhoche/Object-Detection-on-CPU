# Robotics Project Course Submission
The main goal of this project was to make a robot which can:

1. Explore and map its environment
2. Grip and rescue/ bring back the most valuable object it found during 1. 

The robot is equipped with an RGBD camera, a 2-D RP Lidar and among other things, wheel encoders, speaker, arduino (for gripping). It is powered by an intel NUC computer. 

The robot successfully achieves above goals through following pipelines:
- Object (small colorful shapes) detection using image based deep learning.
- Obstacle (heavy batteries and missing walls) detection using image analysis.
- whenever it finds an object/obstacle it speaks out.
- Localization using particle filter.
- Path planning and control using A star to traverse from point A to B.
- Integration is done using a state machine (smach package in ROS) 

If you're interested to know more on how these pipelines are integrated in ROS, check out our [Project page](https://github.com/RAS-2018-grp-4). 

## Videos
Autonomous Exploration             |  What Robot sees during Exploration 
:-------------------------:|:-------------------------:
[![IMAGE ALT TEXT HERE](http://img.youtube.com/vi/khB9kPSdTlk/0.jpg)](https://youtu.be/khB9kPSdTlk)  |  [![IMAGE ALT TEXT HERE](http://img.youtube.com/vi/7GNrhhASpx8/0.jpg)](https://youtu.be/7GNrhhASpx8)

## Contributors: 
- Viktor Tuul (https://www.linkedin.com/in/viktor-tu...)
- Shuo Zheng (https://www.linkedin.com/in/shuo-zheng)
- David Villagra (https://www.linkedin.com/in/davidvill...)
- Ajinkya Khoche (https://www.linkedin.com/in/ajinkyakh...)


