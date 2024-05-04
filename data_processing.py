# -*- coding: utf-8 -*-
"""
data_processing.py

This script contains functions for processing frames captured from a video feed using OpenCV
and landmarks using Mediapipe library. It provides functions for detecting keypoints in the frame, 
drawing landmarks on the frame, collecting keypoints data, and displaying the processed frame.

@author: Pei Yu Chou
"""

import numpy as np
import mediapipe as mp
import cv2


mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities
mp_model = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)


def detect_keypoints(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    image.flags.writeable = False                  # for quicker response
    results = model.process(image)                 
    image.flags.writeable = True                   
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
    draw_landmarks(image, results)
    
    return image, results


def draw_landmarks(image, results):
    drawing_specs = [
        (results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, (80,120,10), (80,256,121), 1, 1),
        (results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, (80,0,10), (80,0,121), 2, 1),
        (results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, (245,117,66), (245,66,20), 4, 2),
        (results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, (245,117,66), (245,66,20), 4, 2)
    ]
    for landmarks, connections, color1, color2, thickness, radius in drawing_specs:
        if landmarks is not None:
            mp_drawing.draw_landmarks(image, landmarks, connections, 
                mp_drawing.DrawingSpec(color=color1, thickness=thickness, circle_radius=radius), 
                mp_drawing.DrawingSpec(color=color2, thickness=thickness, circle_radius=radius))
            

def collect_keypoints(results):
    keypoints = [
        np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3),
        np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4),
        np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3),
        np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    ]
    return np.concatenate(keypoints)


def capure_and_feed_frames(cap, frame_text=None):
    """Capture a frame, perform detection and drawing, finally display it."""
    _, frame = cap.read()
    image, results = detect_keypoints(frame, mp_model)
    draw_landmarks(image, results)
    
    cv2.putText(image, frame_text, (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imshow('OpenCV Frame', image)
    return results