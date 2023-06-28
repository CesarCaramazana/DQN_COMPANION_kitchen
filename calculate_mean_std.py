#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 12:46:19 2023

@author: cczarzosa
"""

import numpy as np
import pickle
import random
import glob
import os
from statistics import mean

"""
even = list([0, 1] * 22)
odd = list([1, 0] * 22)
print(even)


MEAN_VWM_EVEN = -0.1344934
STD_VWM_EVEN = 0.68666816

MEAN_VWM_ODD = 0.18949991
STD_VWM_ODD = 0.652777   


mean = MEAN_VWM_EVEN * np.array(even) + MEAN_VWM_ODD * np.array(odd)
std = STD_VWM_EVEN * np.array(even) + STD_VWM_ODD * np.array(odd)


print("MEAN: ", len(mean))
print("STD: ", std)
"""

root = "./video_annotations/dataset/*"
# labels_pkl = 'labels_updated.pkl'
        
videos = glob.glob(root)

ac_pred = []
ac_rec = []
zs = []

vwm_par = []
vwm_impar = []

# print(len(videos))

num_videos = len(videos)
for video in videos:
    print(video)    
    frames = [filename for filename in os.listdir(video) if filename.startswith("frame")]


    for frame in frames:
        path_frame = os.path.join(video, frame)
        print("Path frame: ", path_frame)
        
        data = np.load(path_frame, allow_pickle=True)
        
        action_prediction = data['data'][0:33]
        action_recognition = data['data'][33:66]
        
        vwm = data['data'][66:110]        
        
        vwm_odd = vwm[1::2]
        vwm_even = vwm[0::2]
        
        vwm_par.append(vwm_even)
        vwm_impar.append(vwm_odd)
        
        z = data['z']
        
        ac_pred.append(action_prediction)
        ac_rec.append(action_recognition)
        zs.append(z)
        

mean = np.mean(np.array(vwm_par))
std = np.std(np.array(vwm_par))

print("Mean VWM par: ", mean)
print("STD VWM par: ", std)

mean = np.mean(np.array(vwm_impar))
std = np.std(np.array(vwm_impar))

print("Mean VWM impar: ", mean)
print("STD VWM impar: ", std)
        
mean_ac_pred = np.mean(np.array(ac_pred))      
std_ac_pred = np.std(np.array(ac_pred))

print("Mean Ac pred: ", mean_ac_pred)
print("STd Ac Pred: ", std_ac_pred)

mean_ac_rec = np.mean(np.array(ac_rec))      
std_ac_rec = np.std(np.array(ac_rec))

print("Mean Ac rec: ", mean_ac_rec)
print("STd Ac rec: ", std_ac_rec)

mean_z= np.mean(np.array(zs))      
std_z= np.std(np.array(zs))

print("Mean Z: ", mean_z)
print("STd Z: ", std_z)


