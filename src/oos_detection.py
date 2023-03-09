#!/usr/bin/env python3
import os
import rospy
from std_msgs.msg import Bool
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
# from detector import Detector
import ros_numpy

# DET_PATH=os.path.join(os.path.dirname(__file__),'efficientdet_d1_coco17_tpu-32')
# detector = Detector(DET_PATH)

active = True
rospy.init_node('oos_detection')
pub = rospy.Publisher('/detection', Detection2DArray, queue_size=2)


def rcv_image(msg):
    global active
    image = ros_numpy.numpify(msg)
    if active:
        detect_and_publish(msg, image)


def detect_and_publish(msg, image):
    global detector

    # detections = detector(image)
    message = Detection2DArray()
    # for clabel,score,box in zip(detections['detection_classes'], detections['detection_scores'], detections['detection_boxes']):
    # d = Detection2D()
    # d.bbox.size_x = box[3]-box[1]
    # d.bbox.size_y = box[2]-box[0]
    # d.bbox.center.x = box[1]+d.bbox.size_x/2
    # d.bbox.center.y = box[0]+d.bbox.size_y/2
    # o = ObjectHypothesisWithPose()
    # o.score = score
    # o.id = clabel
    # d.results.append(o)
    # message.detections.append(d)
    for i in range(1):
        d = Detection2D()
        d.bbox.size_x = 100
        d.bbox.size_y = 100
        d.bbox.center.x = 300
        d.bbox.center.y = 300
        if i == 0: d.source_img = msg
        o = ObjectHypothesisWithPose()
        o.score = 97
        o.id = 0
        d.results.append(o)
        message.detections.append(d)

    pub.publish(message)


def activation_function(msg):
    global active
    active = msg.data


si = rospy.Subscriber("/image_oos", Image, rcv_image)
receiver = rospy.Subscriber("/active_detection", Bool, activation_function)

try:
    rospy.spin()

except KeyboardInterrupt:
    print("Shutting down")
