# -*- coding: utf-8 -*-
"""
action_data_generator.py

This script is designed to generate sequences of keypoint data for action detection. 
It captures frames from a video feed using OpenCV, detects keypoints using the Mediapipe library, 
and collects the keypoint data for a specified action into sequences. 
The collected data can be further utilized for action detection or analysis.

@author: Pei Yu Chou
"""

import os
import cv2
import pandas as pd
from utils import print_args, print_error_in_red
from data_processing import collect_keypoints, capure_and_feed_frames
import argparse


def get_args_parser():
    parser = argparse.ArgumentParser('Generate Sequence of frame for action detection', add_help=False)
    parser.add_argument('label',type=str,
                        help='Name of action for detection')
    parser.add_argument('-o','--output_dir', default='data/', type=str,
                        help='Output path')
    parser.add_argument('-n','--num_sequences', default=100, type=int,
                        help='Number of sequence to collect')
    parser.add_argument('-f','--frames_per_sequence', default=20, type=int,
                        help='Number of frame in each sequence')
    parser.add_argument('-m','--margin', default=5, type=int,
                        help='Number of frame to exclude from the beginning and end of the sequence')
    
    return parser


def main(args):
    print_args(args)
    collection = Action_Collection(args)
    collection.run()


class Action_Collection():
    def __init__(self, args):
        self.label = args.label
        self.output_dir = args.output_dir
        self.total_length = args.frames_per_sequence*args.num_sequences + args.margin*2
        self.margin = args.margin
        self.keypoint_features = []
        
    def run(self):
        self.capture()
        if self.check_collection_complete():
            self.save_csv()
            
    def check_collection_complete(self):
        return len(self.keypoint_features) >= self.total_length
    
    def collect_sequence(self, cap):
        """Collect a sequence of frames for a specific action."""
        while not self.check_collection_complete():
            frame_text = f'Collecting sequence for {self.label} {len(self.keypoint_features)}/{self.total_length} frames...'
            results = capure_and_feed_frames(cap, frame_text=frame_text)
            keypoints = collect_keypoints(results)
            self.keypoint_features.append(keypoints)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
    
    def capture(self):
        cap = cv2.VideoCapture(0)
        self.keypoint_features = []
        waiting_text = f'Press "s" to start collecting for {self.label}'
        try:
            while True:
                key = cv2.waitKey(1)
                capure_and_feed_frames(cap, frame_text=waiting_text)
                # Press "q" to quit
                if key == ord('q'):
                    break
                # Press "s" to start collecting
                if key == ord('s'):
                    self.collect_sequence(cap)
                    break
        finally:
            if cap.isOpened():
                cap.release()
                cv2.destroyAllWindows()
                
    def generate_features_name(self):
        holistic_features = {'face': {'landmark_num': 468,'features': ['x', 'y', 'z']},       
                           'pose': {'landmark_num': 33, 'features': ['x', 'y', 'z', 'visibility']},
                           'lefthand': {'landmark_num': 21,'features': ['x', 'y', 'z']}, 
                           'righthand': {'landmark_num': 21,'features': ['x', 'y', 'z']}, 
                          }
        features_name = []
        for body_part, body_part_dict in holistic_features.items():
            landmark_num, features = body_part_dict['landmark_num'], body_part_dict['features']
            for idx_landmark in range(1, landmark_num+1):
                for ft in features:
                    features_name.append(f"{body_part}_landmark{idx_landmark}_{ft}")
                    
        return features_name
    
    def save_csv(self):  
        features_name = self.generate_features_name()
        df = pd.DataFrame(self.keypoint_features[self.margin:-self.margin], columns=features_name)
        df = df.assign(label=self.label)
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            output_file = os.path.join(self.output_dir, f'{self.label}.csv')
            df.to_csv(output_file, index=False) 
            print(f'Finish collecting, saved results to {output_file}')
        except OSError as e:
            print(f'Error:Failed to save output file: {e}')
    
    
if __name__ == '__main__':
    try:
        parser = get_args_parser()
        args = parser.parse_args()
        main(args)
    except Exception as error:
        print_error_in_red(f'{type(error).__name__} - {error}')
        raise error
  