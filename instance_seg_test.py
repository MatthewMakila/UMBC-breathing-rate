# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 13:17:18 2023

@author: matth
"""
#%% Importing ...

# prevent multithread use by other applications
import os
import glob
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import pixellib
from pixellib.torchbackend.instance import instanceSegmentation
import cv2
import matplotlib.pyplot as plt
print("Success!") 

#%% Perform segmentation(s)

capture = cv2.VideoCapture(0)

# press q to exit

segment_video = instanceSegmentation()
segment_video.load_model("pointrend_resnet50.pkl", confidence=0.7, detection_speed="rapid")
segment_video.process_camera(capture,  show_bboxes = True, frames_per_second= 30, check_fps=True, show_frames= True,
frame_name= "frame", output_video_name="output_video.mp4")
captureq.release()
cv2.destroyAllWindows()

#ins.segmentImage("person_walking_far_sample2.jpg", show_bboxes=True, output_image_name="output_image2.jpg")

#%% Outputs

# Display image(s)
im = cv2.imread("output_image2.jpg")
im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
plt.imshow(im_rgb)

# Extract instance from background, apply neutral background
""""""

# Then test on our video ...
path_dir = 'data_/'
dataPath = os.path.join(path_dir, '*.mp4')
files = glob.glob(dataPath)  # care about the serialization
list.sort(files) # serialing the data

# check files loaded
if not files:
    raise Exception("Data upload failure!")
    
# print files list
print(*(x for x in files), sep='\n')
print('\n')

#ins.load_model("pointrend_resnet50.pkl", detection_speed = "rapid")
#ins.process_video(files[1], show_bboxes=True, frames_per_second=30, output_video_name="output_video_1.mp4")

