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