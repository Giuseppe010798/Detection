#!/usr/bin/env python3

import os
import time

import numpy as np
import numpy.random
import ros_numpy
import rospy
import tensorflow as tf
from sensor_msgs.msg import Image, CompressedImage
from vision_msgs.msg import Detection2D, ObjectHypothesisWithPose, Detection2DArray

KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

active = True
rospy.init_node('people_detection')

global model


def rcv_image(msg):
    global active
    rospy.loginfo("--- IMAGE RECEIVED ---")
    image = ros_numpy.numpify(msg)
    # --- Normal Use ---
    if active:
        detect_and_publish(msg, image)
    # --- Inference Use ---
    """
    image = msg
    if active:
        # tic = time.perf_counter()
        detect_and_publish(msg, image)
        # toc = time.perf_counter()
        # print(f"Time {toc - tic:0.4f} seconds")
        # return toc - tic
    """


def to_pixel_coords(left_obj, right_obj):
    left_obj_pixel_cord = (left_obj[0], left_obj[1])
    right_obj_pixel_cord = (right_obj[0], right_obj[1])

    left_pixel_cord = tuple(round(coord * dimension) for coord, dimension in zip(left_obj_pixel_cord, (640, 480)))
    right_pixel_cord = tuple(round(coord * dimension) for coord, dimension in zip(right_obj_pixel_cord, (640, 480)))

    return left_pixel_cord, right_pixel_cord


def get_body_piece(left_index, right_index, keypoints):
    left_x = float(keypoints[0][0][left_index][0])
    left_y = float(keypoints[0][0][left_index][1])
    left = (left_x, left_y)

    right_x = float(keypoints[0][0][right_index][0])
    right_y = float(keypoints[0][0][right_index][1])
    right = (right_x, right_y)

    left_pixel, right_pixel = to_pixel_coords(left, right)

    score_piece = float(keypoints[0][0][left_index][2]) + float(keypoints[0][0][right_index][2])

    return left_pixel, right_pixel, score_piece


def run_inference_for_single_image(image):
    image = np.asarray(image)
    output_dict = model(image)
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections
    output_dict['detection_classes'] = output_dict[
        'detection_classes'].astype(np.int64)
    return output_dict


def show_inference(image, class_id=1):
    image_np = np.array(image)
    output_dict = run_inference_for_single_image(image_np)
    boxes = []
    classes = []
    scores = []
    for i, x in enumerate(output_dict['detection_classes']):
        if x == class_id and output_dict['detection_scores'][i] > 0.5:
            classes.append(x)
            boxes.append(output_dict['detection_boxes'][i])
            scores.append(output_dict['detection_scores'][i])
    boxes = np.array(boxes)
    classes = np.array(classes)
    scores = np.array(scores)
    return boxes, classes, scores


def detect_and_publish(msg, image):
    image = tf.expand_dims(image, axis=0)

    # --- MODELLO MobileNetV2 ---

    image = tf.cast(tf.image.resize_with_pad(image, 320, 320),
                    dtype=tf.uint8)

    boxes, classes, scores = show_inference(image)

    max_x = 0
    max_y = 0
    min_x = 0
    min_y = 0
    score = 0

    try:
        if boxes is not None:
            max_x = round(boxes[0][3] * 640)
            max_y = round(boxes[0][2] * 480)
            min_x = round(boxes[0][1] * 640)
            min_y = round(boxes[0][0] * 480)
            score = scores[0]
    except:
        pass

    width = max_x - min_x
    height = max_y - min_y
    center_x, center_y = min_x + width / 2, min_y + height / 2

    # --- MODELLO TF/TF_RT ---
    """
    image = tf.cast(tf.image.resize_with_pad(image, 192, 192),
                    dtype=tf.int32)  # Single pose 192*192, Multi pose 256*256

    movenet = model.signatures['serving_default']
    outputs = movenet(image)
    keypoints = outputs['output_0']

    left_eye, right_eye, eye_score = get_body_piece(1, 2, keypoints)  # Get COORDINATES of EYES in pixel
    left_shoulder, right_shoulder, shoulder_score = get_body_piece(5, 6,
                                                                   keypoints)  # Get COORDINATES of SHOULDERS in pixel
    # left_knee, right_knee, knee_score = get_body_piece(13, 14, keypoints)  # Get COORDINATES of KNEES in pixel

    score = eye_score + shoulder_score  # + knee_score

    coords = (left_shoulder, right_shoulder)

    # print("--- EYES PIXEL COORDINATES: ", str(left_eye) + str(right_eye) + " ---")
    # print("--- SHOULDER PIXEL COORDINATES: ", str(left_shoulder) + str(right_shoulder) + " ---")
    # print("--- KNEE PIXEL COORDINATES: ", str(left_knee) + str(right_knee) + " ---")

    minx, miny = np.min(coords, axis=0)
    maxx, maxy = np.max(coords, axis=0)

    width = maxx - minx
    height = maxy - miny
    center_x, center_y = minx + width / 2, miny + height / 2
    """

    message = Detection2DArray()
    d = Detection2D()
    d.bbox.size_x = max_x
    d.bbox.size_y = max_y
    d.bbox.center.x = center_x
    d.bbox.center.y = center_y

    # --- Rescaling image from 640x480 to 320x240 ---
    # np_image = ros_numpy.msgify(Image, np_image[::2, ::2], encoding="rgb8")
    d.source_img = msg

    o = ObjectHypothesisWithPose()
    o.score = score
    o.id = 0
    d.results.append(o)
    message.detections.append(d)
    pub.publish(message)


try:

    rospy.loginfo("--- LOADING MODEL ---")

    DET_PATH = os.path.join(os.path.dirname(__file__),
                            'MobileNetV2')

    # --- MODELLO TF/TF_RT ---
    model = tf.saved_model.load(DET_PATH)

    rospy.loginfo("--- MODEL LOADED ---")

    pub = rospy.Publisher('/detection', Detection2DArray, queue_size=1)
    si = rospy.Subscriber("/image_people", Image, rcv_image)

    # --- Inference ---
    """
    inference_time = 0

    for i in range(0, 1000):
        matrix = np.random.randint(0, 255, (480, 640, 3))
        inference_time += rcv_image(matrix)

    print("Inference time: ", inference_time / 1000)
    """

    rospy.spin()

except KeyboardInterrupt:
    print("Shutting down")
