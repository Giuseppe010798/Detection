#!/usr/bin/python3

import os
import time
import rospy
import cv2
from sensor_msgs.msg import Image
import ros_numpy

os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'

DEBUG = True
RELEASE = False

rospy.init_node("camera_acquisition")

# cam_oos = cv2.VideoCapture(0)
cam_people = cv2.VideoCapture(0)

# image_publisher_oos = rospy.Publisher('/image_oos', Image, queue_size=1)
image_publisher_people = rospy.Publisher('/image_people', Image, queue_size=1)

while not rospy.is_shutdown():

    # ret_oos, frame_oos = cam_oos.read()
    ret_people, frame_people = cam_people.read()

    # if ret_oos:
    #    msg = ros_numpy.msgify(Image, frame_oos, encoding='rgb8')
    #    image_publisher_oos.publish(msg)

    if ret_people:
        msg = ros_numpy.msgify(Image, frame_people, encoding='rgb8')
        image_publisher_people.publish(msg)

    time.sleep(0.5)

cam_people.release()
