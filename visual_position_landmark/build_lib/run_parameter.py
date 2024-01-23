#!/usr/bin/python3

import argparse
import cv2
import os

def get_input():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_mode', type = str, default="image")
    parser.add_argument("--save_path",type = str, default= "./")
    args = parser.parse_args()
    return args


def tat_chuong_trinh(sig, frame):
    print("Ctrl C . Exiting....... ")
    exit_flag = True

def check_array(actual_position):
    run = 0
    for row in actual_position:
        for element in row:
            if element !=0 : 
                run = 1
    return run
def save_video():
    if os.path.exists(os.path.join(os.getcwd(), "video_1.mp4")):
        os.remove(os.path.join(os.getcwd(), "video_1.mp4"))
    else:
        width = 848
        height = 480
        fps = 15
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(os.path.join(os.getcwd(), "video_1.mp4"), fourcc, fps, (width, height))
    return out