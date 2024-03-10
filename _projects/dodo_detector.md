---
layout: page
title: dodo detector
description: Object detection for robotics with Python and ROS
# img: assets/img/12.jpg
importance: 2
category: computer vision
related_publications: true
---

During my stay at FEI University Center, I was responsible for integrating object detection techniques into a domestic robot. I worked with libraries such as OpenCV and the TensorFlow Object Detection API as well as ROS, the Robot Operating System. This section provides a centralized collection of resources I've created during my years working with computer vision and object detection.

Since you're here, take a look at the [set of Python scripts](https://github.com/douglasrizzo/detection_util_scripts) I provide to help in the creation of TFRecord files and label maps for the TensorFlow Object Detection API, as well as TXT annotation files for YOLOv3. I also authored a brief tutorial on how to train a detection model using the TensorFlow Object Detection API with the help of my scripts.

### dodo detector (Python)

[dodo detector](http://douglasrizzo.com.br/dodo_detector/) is a Python package that encapsulates OpenCV object detection via keypoint detection and feature matching, as well as the [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) in a single place. See a simple tutorial [here](https://gist.github.com/douglasrizzo/fd4cff7cdf53b3ad08d67f736e5017ea). [[Relevant paper]](https://www.researchgate.net/publication/338032150_CAPTION_Correction_by_Analyses_POS-Tagging_and_Interpretation_of_Objects_using_only_Nouns).

<iframe width="560" height="315" src="https://www.youtube.com/embed/Py6_qG52EYQ" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

### dodo detector ros (ROS)

I am also the creator of [dodo detector ros](https://github.com/douglasrizzo/dodo_detector_ros), a ROS package that allows dodo detector to interface with USB cameras as well as both Kinects v1 and v2, in order to detect objects and place them in a 3D environment using the Kinect point cloud. [[Relevant paper]](https://www.researchgate.net/publication/333931333_HERA_Home_Environment_Robot_Assistant?_sg=GeiJpHAg-qFfldnKYUJofw09SmBojDPMoOVXAXBtRN0PQoe-1N-CM7ry2q89Gq0zfcwUusFYgBCG1U3dN-KoIGfndqnR9tazsZ9_gafb.7OO3N70IPnsb377if8wOMVhPMKJnucTmYH7hn34kpeBcKn_KwIOVF1m28fGLgwgO06jL6mvZR1RcBnDIYMAvwQ).

<iframe width="560" height="315" src="https://www.youtube.com/embed/fXJYmJOaSxQ" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
