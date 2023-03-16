#!/usr/bin/env python3

import ros_numpy
import rospy
from move_base_msgs.msg import MoveBaseGoal
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
from std_msgs.msg import Int8

rospy.init_node('merging_node')

center_x = 0
center_y = 0
image = None
score = 0


def rcv_detection(msg: Detection2DArray):
    global center_x
    global center_y
    global image
    global score

    image = ros_numpy.numpify(msg.detections[0].source_img)
    score = msg.detections[0].results[0].score

    if image is None:
        return
    for d in msg.detections:
        center_x = d.bbox.center.x
        center_y = d.bbox.center.y


def convert_depth_image(msg: Image):
    im = ros_numpy.numpify(msg)
    print('Depth at center(%d, %d): %f(mm)\r' % (int(center_x), int(center_y), im[int(center_x)][int(center_y)]))
    goal_msg = MoveBaseGoal()
    goal_msg.target_pose.header.frame_id = "base_scan"
    goal_msg.target_pose.header.stamp = rospy.Time.now()
    goal_msg.target_pose.pose.position.x = float(im[int(center_x)][int(center_y)] / 1000)
    goal_msg.target_pose.pose.position.y = 0
    goal_msg.target_pose.pose.orientation.x = 0
    goal_msg.target_pose.pose.orientation.y = 0
    goal_msg.target_pose.pose.orientation.z = 0
    goal_msg.target_pose.pose.orientation.w = 1
    pub.publish(goal_msg)
    """
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(msg, msg.encoding)
    pix = (msg.width / 2, msg.height / 2)
    # print('Depth at center(%d, %d): %f(mm)\r' % (int(pix[0]), int(pix[1]), cv_image[int(pix[1]), int(pix[0])]))
    print('Depth at center(%d, %d): %f(mm)\r' % (int(center_x), int(center_y), cv_image[int(center_y), int(center_x)]))
    """
    """
    if score >= 0.50:
        bridge = CvBridge()
        # Use cv_bridge() to convert the ROS image to OpenCV format
        # Convert the depth image using the default passthrough encoding
        depth_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        depth_array = np.array(depth_image, dtype=np.float32)
        center_idx = np.array(depth_array.shape) / 2
        print('center depth:', depth_array[int(center_x), int(center_y)])
    else:
        pass
    """


rs_sub = rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, convert_depth_image, buff_size=1)
img_sub = rospy.Subscriber('/detection', Detection2DArray, rcv_detection, queue_size=1)
pub = rospy.Publisher('/depth', MoveBaseGoal, queue_size=10)

try:
    rospy.spin()

except KeyboardInterrupt:
    print("Shutting down")
