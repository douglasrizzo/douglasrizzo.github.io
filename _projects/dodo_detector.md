---
layout: page
title: dodo detector
description: Object detection for robotics with Python and ROS
# img: assets/img/12.jpg
importance: 1
category: computer vision
tags: convnets open-source object-detection tensorflow ros
related_publications: true
---

During my stay at FEI University Center, I was responsible for integrating object detection techniques into a domestic robot. I worked with libraries such as OpenCV and the TensorFlow Object Detection API as well as ROS, the Robot Operating System. This section provides a centralized collection of resources I've created during my years working with computer vision and object detection.

Since you're here, take a look at the [set of Python scripts](https://github.com/douglasrizzo/detection_util_scripts) I provide to help in the creation of TFRecord files and label maps for the TensorFlow Object Detection API, as well as TXT annotation files for YOLOv3. I also authored a brief tutorial on how to train a detection model using the TensorFlow Object Detection API with the help of my scripts.

### dodo detector (Python)

[dodo detector](http://douglasrizzo.com.br/dodo_detector/) is a Python package that encapsulates OpenCV object detection via keypoint detection and feature matching, as well as the [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) in a single place. See a simple tutorial [here](https://gist.github.com/douglasrizzo/fd4cff7cdf53b3ad08d67f736e5017ea). This code was used in the experiments for the paper {% cite Ferreira2022 %}.

<iframe width="560" height="315" src="https://www.youtube.com/embed/Py6_qG52EYQ" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

### dodo detector ros (ROS)

I am also the creator of [dodo detector ros](https://github.com/douglasrizzo/dodo_detector_ros), a ROS package that allows dodo detector to interface with USB cameras as well as both Kinects v1 and v2, in order to detect objects and place them in a 3D environment using the Kinect point cloud. This code was used to develop Hera {% cite AquinoJunior2019 %}, a domestic robot that won the 2023 RoboCup, as well as several Latin American robotics competitions.

<iframe width="560" height="315" src="https://www.youtube.com/embed/fXJYmJOaSxQ" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
