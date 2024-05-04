# -*- coding: utf-8 -*-
"""
Real-time Action Detection

This script utilizes a pre-trained model for real-time action detection. 
It combines MediaPipe for landmark detection and a LSTM for action prediction. 
Run the script with the path to the detection model checkpoint.
View the real-time action detection on the screen and press "q" to quit the application.

@author: Pei Yu Chou
"""

import cv2
import numpy as np
import torch
import mediapipe as mp
from data_processing import detect_keypoints, draw_landmarks, collect_keypoints
from model import LSTMModel
import argparse


def get_args_parser():
    parser = argparse.ArgumentParser('Sign detection by realtime testing', add_help=False)
    parser.add_argument('checkpoint', type=str,
                        help='Path of detection model')
    return parser

def main(args):
    model = Action_Detection_Model(args.checkpoint)
    realtime_testing(model)
    
def realtime_testing(model):
    mp_holistic = mp.solutions.holistic # Holistic model
    cap = cv2.VideoCapture(0)
    seqence = []
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        try:
            while True:
                key = cv2.waitKey(1)
                ret, frame = cap.read()
                               
                # Make detections and Draw
                image, results = detect_keypoints(frame, holistic)
                draw_landmarks(image, results)
                # results = capure_and_feed_frames(cap, frame_text=None)
                keypoints = collect_keypoints(results)
                seqence.append(keypoints)
                
                if len(seqence)>=10:
                    prediction = model.predict(np.array(seqence))
                    cv2.putText(image, f'{prediction}', (280,30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    
                cv2.putText(image, 'press "q" to quit', (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.imshow('Action Detection', image)
                # Press "q" to quit
                if key == ord('q'):
                    break
               
        finally:
            if cap.isOpened():
                cap.release()
                cv2.destroyAllWindows()
            

class Action_Detection_Model():
    def __init__(self, checkpoint_path):
        self.checkpoint_path = checkpoint_path
        self.load()
        
    def load(self):
        checkpoint = torch.load(self.checkpoint_path)
        self.model_param = checkpoint["model_param"]
        
        self.model = LSTMModel(input_size=self.model_param["input_size"], 
                          hidden_size=self.model_param['hidden_size'], 
                          num_layers=self.model_param['num_layers'], 
                          num_classes=self.model_param['num_classes'])
        self.model.load_state_dict(checkpoint["model"])
        self.classes = list(self.model_param["classes"].keys())
        self.seqence_length =  self.model_param["seqence_length"]
        
    def predict(self, seqence):
        inputs = seqence[-self.seqence_length:]
        output = self.model(torch.tensor(inputs, dtype=torch.float32).unsqueeze(dim=0))
        _, prediction = torch.max(output, 1)
        return self.classes[prediction.item()]
        
    
if __name__ == '__main__':
    try:
        parser = get_args_parser()
        args = parser.parse_args()        
        main(args)
    except Exception as error:
        raise error
