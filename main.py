import open3d as o3d
import time
import cv2
# from tracking import pipeline as tracker
import os
from os.path import exists
from reconstruction.utility.file import make_clean_folder, get_rgbd_file_list2
import numpy as np
from dfu_detector.detect_frcnn_dfu import Faster_RCNN
from tracker.tracker import Tracker, BoundingBox
import json
from cropping.cropping import cropp_the_images_with_stepped_recognition, rectifie, create_ulcer_rec
from cropping.save import save_all_data_cropped
from reconstruction.pipeline import pipeline as reconstruction_pipeline

LEFT_ARROW = 2424832
RIGHT_ARROW = 2555904
SCAPE = 27
SPACE_BAR = 32

# ignore TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)


def save_frames(bag_path, output_path):
    bag_reader = o3d.t.io.RSBagReader()
    bag_reader.open(bag_path)
    bag_reader.save_frames(output_path)


if __name__ == "__main__":
    config = {
        "path_dataset": 'C:\\Users\\53588\\Desktop\\Tesis\\CurrentProject\\dfu_thesis FORK\\dfu_thesis\\output',
        "frame_step": 3
    }

    # # Get frames
    # bag_path = 'C:\\Users\\53588\\Desktop\\Tesis\\bags uncompressed\\20230511_102553.bag'
    # save_frames(bag_path,'output')

    # # Initialize tracker
    # tracker = Tracker()
    # # Start Cropp
    # data = cropp_the_images_with_stepped_recognition(
    # config, tracker, create_ulcer_rec(), 10, rectifie)
    # # Save image cropped
    # save_all_data_cropped(data)

    

    reconstruction_pipeline({})
