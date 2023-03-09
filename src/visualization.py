#!/usr/bin/env python3

import ros_numpy
import rospy
from vision_msgs.msg import Detection2DArray
import cv2

rospy.init_node('visualization_node')


def rcv_detection(msg):
    rospy.loginfo('--- DETECTION HERE ---')
    im = ros_numpy.numpify(msg.detections[0].source_img)
    if im is None:
        return
    h, w, _ = im.shape
    for d in msg.detections:
        b = [d.bbox.center.y, d.bbox.center.x, d.bbox.size_y, d.bbox.size_x]
        b[0] -= b[2] / 2
        b[1] -= b[3] / 2
        start_point = (int(b[1]), int(b[0]))
        end_point = (int(b[3]), int(b[2]))
        col = (0, 255, 0)
        cv2.rectangle(im, start_point, end_point, col, 3)
        # p1 = (p1[0] - 10, p1[1])
        # cv2.putText(im, "%s %.2f" % (c, s), p1, cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
    cv2.imshow('Image', im)
    cv2.waitKey(100)


sd = rospy.Subscriber("/detection", Detection2DArray, rcv_detection, buff_size=2 ** 28)

try:
    rospy.spin()

except KeyboardInterrupt:
    print("Shutting down")
