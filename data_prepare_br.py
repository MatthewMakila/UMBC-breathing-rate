#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 23:52:27 2023

@author: zahid

"""
#%% libraries
import os

import matplotlib.pyplot as plt

import numpy as np

import cv2
import cv2 as cv

import glob

from scipy.io import loadmat
from scipy.signal import butter, filtfilt, buttord
from scipy.fft import fft, fftfreq

import random

from random import seed, randint

from sys import exit

# from sklearn.model_selection import train_test_split

import pandas as pd
import pickle
import pdb

#%%  Data Load files from the directory

# Select the source file [either MERL or Collected Dataset or ]


# load Pathdir
#iD_ir = '../../../Dataset/Merl_Tim/Subject1_still/IR'
#iD_ir = '../../../Dataset/Merl_Tim/Subject1_still/RGB_raw'
#iD_ir = '../../../Dataset/Merl_Tim/Subject1_still/RGB_demosaiced'

path_dir = 'data_/'

dataPath = os.path.join(path_dir, '*.mp4')

files = glob.glob(dataPath)  # care about the serialization
# end load pathdir
list.sort(files) # serialing the data

if not files:
    raise Exception("Data upload failure!")

# Take time stamp and multiple by 64. Take starting time of the BVP file, 
# subtract the tags.csv from the BVP start time, multiply by 64 to get the sample number. 


#%% Load the Video and corresponding

# find start position by pressing the key position in empatica
# perfect alignment! checked by (time_eM_last-time_eM_first)*30+start_press_sample  should
# give the end press in video


def data_read(files, im_size = (200, 200)):
    data = []
    cap = cv2.VideoCapture(files[0])
    
    valss= np.zeros((200,200,3))
    
    # import pdb
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        
        if ret==False:
            break
    
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        gray = gray[:,:,:]
        gray = gray[100:, 100:1400]
        gray = cv2.resize(gray, im_size)
        # pdb.set_trace()
        data.append(gray)

        cv2.imshow('frame', gray)
        valss = gray
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    
    # fps = cap.get(cv2.CAP_PROP_FPS)
        
    cap.release()
    cv2.destroyAllWindows()
    data =  np.array(data)
    
    return data

def data_read_of(files, im_size = (200, 200)):
    data = []
    cap = cv2.VideoCapture(files[0])
    ret, frame1 = cap.read()
    frame1 = cv2.resize(frame1, im_size)
    prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255
    while(1):
        ret, frame2 = cap.read()
        frame2 = cv2.resize(frame2, im_size)
        if not ret:
            print('No frames grabbed!')
            break
        next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang*180/np.pi/2
        hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
        bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        cv.imshow('frame2', bgr)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv.imwrite('opticalfb.png', frame2)
            cv.imwrite('opticalhsv.png', bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        prvs = next
    cv.destroyAllWindows()
    return data

def test_video(files, im_size=(200, 200)):
    # begin video capture
    f_num = 22
    print(files[f_num])
    cap = cv2.VideoCapture(files[f_num])
        
    ijk = 0

    while(cap.isOpened()):
        ret, frame1 = cap.read()
        
        if ret==False:
            break
            
        # pdb.set_trace()
        if (ijk == 2000):
              breakpoint()  
        ijk += 1
        
        frame1 = frame1[:,:,:]
        frame1 = cv2.resize(frame1, im_size)

        img_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(img_gray, (3,3), 0) 
         
        # Canny Edge Detection
        edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=100) # Canny Edge Detection
        
        # Display Canny Edge Detection Image
        cv2.imshow('Canny Edge Detection', edges)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    
def data_read_ed(files, im_size=[200, 200]):
    data = []
    data_end = len(files)
    
    # print files list
    print(*(x for x in files), sep='\n')
    print('\n')
    
    # take file input and handle issues
    try:
        f_num = int(input("Video Sample Request: "))
        if (f_num < 0 or f_num > data_end):
            raise Exception("Bad Video Request ...")
            
    except:
        raise Exception("Bad Video Request ...")
    
    # bank of manual annotations for video sample ROIs 
    # (sample no. : [[rows], [cols]])
    """BAD BANK :(      [21]"""
    annotation_bank = {1: [[500, 750], [250, 750]], 2: [[500, 750], [250, 750]], 
                       3: [[350, 650], [200, 750]], 4: [[500, 750], [350, 800]],
                       5: [[1100, 1600], [400, 1000]], 6: [[0, 150], [500, 550]],
                       7: [[550, 700], [100, 400]], 8: [[850, 1200], [250, 900]],
                       9: [[1100, 1400], [0, 600]], 10: [[450, 650], [0, 350]],
                       11: [[800, 1000], [350, 1000]], 12: [[1200, 1450], [0, 600]],
                       13: [[400, 550], [100, 300]], 14: [[1000, 1300], [400, 1000]],
                       15: [[275, 400], [150, 400]], 16: [[300, 450], [50, 250]],
                       17: [[550, 650], [100, 350]], 18: [[1100, 1400], [150, 900]],
                       19: [[1200, 1400], [300, 800]], 20: [[1000, 1300], [0, 200]],
                       21: [[0, 0], [0, 0]], 22: [[1100, 1250], [500, 650]], 
                       23: [[1100, 1300], [1500, 1900]], 24: [[650, 750], [1400, 1600]],
                       25: [[750, 900], [900, 1000]], 26: [[600, 850], [1200, 1350]],
                       27: [[500, 650], [600, 850]], 28: [[550, 700], [450, 600]]}
    
    # begin video capture
    cap = cv2.VideoCapture(files[f_num])
    frame_num = 0
    
    valss= np.zeros((200,200,3))
    
    # import pdb
    ijk = 0
    if (ijk == 0):
        print(files[f_num])


    while(cap.isOpened()):
        ret, frame1 = cap.read()
        
        if ret==False:
            break
            
        # pdb.set_trace()
        # if (ijk == 1000):
        #       breakpoint()
            
        # ijk += 1
        
        # take manual annotations for selected video data
        row_1 = annotation_bank[f_num][0][0]
        row_2 = annotation_bank[f_num][0][1]
        col_1 = annotation_bank[f_num][1][0]
        col_2 = annotation_bank[f_num][1][1]
        
        frame1 = frame1[:,:,:]
        frame1 = frame1[row_1:row_2, col_1:col_2] 
        im_size[1] = row_2 - row_1
        im_size[0] = col_2 - col_1
        im_size = tuple(im_size)
        frame1 = cv2.resize(frame1, im_size)
        im_size = list(im_size)

        img_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        
        img_blur = cv2.GaussianBlur(img_gray, (3,3), 0) 
         
        # Canny Edge Detection
        
        # play with threshold values
        
        edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=100) # Canny Edge Detection
        # Display Canny Edge Detection Image
        data.append(edges)
        
        if (frame_num >= 500):
            cv2.imshow('Canny Edge Detection', edges)
        # ijk += 1
        # pdb.set_trace()
        
        frame_num += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    
    # fps = cap.get(cv2.CAP_PROP_FPS)
        
    cap.release()
    cv2.destroyAllWindows()
    data =  np.array(data)
    
    return data

# test_video(files)
data_og = data_read_ed(files)
data = data_og

#%% Filtering functions

# Butter filter req.
fs = 30      # sample rate, Hz
cutoff = 0.3      # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
nyq = 0.5 * fs  # Nyquist Frequency
order = 2       # sin wave can be approx represented as quadratic
n = int(500) # total number of samples

# Buttord filtering
low_cut = 40    # lower freq cutoff
high_cut = 60   # higher freq cutoff
rp = 3  # passband ripple (dB)
rs = 40 # stopband attenuation (dB)

def butter_lowpass_filter(data, cutoff, fs, order):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    # order, normal_cutoff = buttord(0.5, 0.6, 3, 40, analog=False)
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


#%% Analyze data
fraction = 3 # split current video sample length
samp_size_start = data.shape[0] // fraction
timeframe_samp = 2000 # no. of frames examined
samp_rate = 1 # frequency (frame gap, i.e., check every "5" frames)

print(data.shape, samp_size_start, timeframe_samp)

# the edge img
# plt.imshow(data[samp_size_start + i])

data_diff = []
diff_frame = 2
d_range_adjustment = 500
# calculate a data set for data with differences between its nearest 
# neighboring (diff_frame) frames
for i in range(data.shape[0] - d_range_adjustment):
    data_diff.append(data[i + diff_frame] - data[i])
    
data_diff = np.array(data_diff)

def row_col_analyze(data_axis):
    """
    data_axis:  1 for row, 2 for col
    return: np arrays carrying processed data 
    """
    plot_choice_str = ["Row", "Column"]
    
    # row/col sum & diffs between neighboring frames
    data_summed = np.sum(data, axis=data_axis)
    data_summed_diff = np.sum(data_diff, axis=data_axis)
    arr = []
    arr_diff = []

    # append data samples of max row/col/sum val at particular frame
    for i in range(timeframe_samp):
        if ((samp_size_start + i * samp_rate) >= data.shape[0]):
            break
        arr.append(np.argmax(data_summed[samp_size_start + i * samp_rate]))
        arr_diff.append(np.argmax(data_summed_diff[samp_size_start + i * samp_rate]))      

    # convert to np arr
    arr = np.array(arr)
    arr_diff = np.array(arr_diff)

    # perform plotting of og signal and signal taking diff
    fig1, axs1 = plt.subplots(2)
    fig1.suptitle('{}s with Max no. Edge Points vs. Timeframe (in frames)'.format(plot_choice_str[data_axis - 1]))
    axs1[0].plot(arr)
    axs1[0].set(ylabel='{}s with Max no. Edge Points'.format(plot_choice_str[data_axis - 1]))
    axs1[1].plot(arr_diff)
    axs1[1].set(ylabel='diff')
    axs1[1].set(xlabel='Timeframe (in frames)')
    
    return arr, arr_diff

def full_sum_analyze(data_axis):
    """
    data_axis:  1 for row, 2 for col, 0 for full sum
    return: np arrays carrying processed data 
    """
    arr = []
    arr_diff = []

    # append data samples of max sum val at particular frame
    for i in range(timeframe_samp):
        if ((samp_size_start + i * samp_rate) >= data.shape[0]):
            break
        arr.append(np.sum(data[samp_size_start + i * samp_rate]))
        arr_diff.append(np.sum(data_diff[samp_size_start + i * samp_rate]))

    # convert to np arr
    arr = np.array(arr)
    arr_diff = np.array(arr_diff)

    # perform plotting of og signal and signal taking diff
    fig1, axs1 = plt.subplots(2)
    fig1.suptitle('Edge Points Summed in Frame vs. Timeframe (in frames)')
    axs1[0].plot(arr)
    axs1[0].set(ylabel='No. Summed Points')
    axs1[0].set(xlabel='Timeframe (frames)')
    axs1[1].plot(arr_diff)
    axs1[1].set(ylabel='diff')
    axs1[1].set(xlabel='Timeframe (frames)')

    
    return arr, arr_diff

def filt_and_plot(arr, plot_choice):
    """
    arr:    specific arr to plot
    plot_choice:    choice of row, col, or sum plot
    """
    plot_choice_str = ["Sum", "Row", "Column"]
    y = butter_lowpass_filter(arr, cutoff, fs, order)
    plt.figure()
    plt.plot(y)
    plt.title('{}s with Max no. Edge Points vs. Timeframe (in frames)'.format(plot_choice_str[plot_choice]))
    plt.show()

rs_arr, rs_arr_diff = row_col_analyze(1)
cs_arr, cs_arr_diff = row_col_analyze(2)
fs_arr, fs_arr_diff = full_sum_analyze(0)

# do some filtering
"""WORK ON HOW TO SELECT SIGNAL BASED ON FFT FREQ ANALYSIS"""
filt_and_plot(fs_arr, 0)
filt_and_plot(fs_arr_diff, 0)
filt_and_plot(cs_arr, 2)
filt_and_plot(cs_arr_diff, 2)



#%% Load CSV file

exit()

import pandas as pd
data_CSV= pd.read_csv("csv_data/sample_20_sitting(Hars_5_12_23).csv")

## 211 for subject sample 1


#%% Data Cross-Check and Visualization

i = 840
plt.imshow(data[211+i*3])
print(data_CSV['Data Set 1:Force(N)'][i])

data_c = data[211:]

#%% Check frame and force
import time
k = 121
for i in range(10):
    cv2.imshow('frame',data_c[5*3*(i+k)])
    time.sleep(2)
    print(data_CSV['Data Set 1:Force(N)'][5*(i+k)])
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    time.sleep(3)

