#!/usr/bin/env python3
import math

import ros_numpy
import rospy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray


class DepthConverter:
    def __init__(self):
        rospy.init_node('merge_depth_node')
        self.center_x = 0
        self.center_y = 0
        self.depth_sub = rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, self.convert_depth_image,
                                          buff_size=2 ** 28)
        self.img_sub = rospy.Subscriber('/detection', Detection2DArray, self.rcv_detection, queue_size=1)
        self.pub = rospy.Publisher('/depth', PoseStamped, queue_size=10)

    def rcv_detection(self, msg: Detection2DArray):
        image = ros_numpy.numpify(msg.detections[0].source_img)

        if image is None:
            return
        for d in msg.detections:
            # In d.bbox.center there are min_x and min_y instead
            width = d.bbox.size_x - d.bbox.center.x  # max_x - min_x
            height = d.bbox.size_y - d.bbox.center.y  # max_y - min_y
            self.center_x, self.center_y = d.bbox.center.x + width / 2, d.bbox.center.y + height / 2

    def convert_depth_image(self, msg: Image):
        im = ros_numpy.numpify(msg)
        print('Depth at center(%d, %d): %f(mm)\r' % (
            int(self.center_x), int(self.center_y), im[int(self.center_x)][int(self.center_y)]))

        pose = self.get_depth_pose(im)
        self.pub.publish(pose)

    def get_depth_pose(self, im):
        depth_pose = PoseStamped()

        depth_pose.header.frame_id = "base_scan"
        depth_pose.header.stamp = rospy.Time.now()

        depth_pose.pose.position.x = float(im[int(self.center_x)][int(self.center_y)] / 1000)
        depth_pose.pose.position.y = 0
        depth_pose.pose.position.z = 0

        depth_pose.pose.orientation.x = 0
        depth_pose.pose.orientation.y = 0
        depth_pose.pose.orientation.z = 0
        depth_pose.pose.orientation.w = 1

        return depth_pose

    @staticmethod
    def run():
        try:
            rospy.spin()
        except KeyboardInterrupt:
            print("Shutting down")


if __name__ == '__main__':
    DepthConverter().run()
