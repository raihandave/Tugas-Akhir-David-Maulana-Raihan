# -*- coding: utf-8 -*-
"""Untitled39.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1RFFU84aMhFCfh8K2S1kDbaEnwyf3ox6_
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

def ORB():
    root = os.getcwd()
    imgPath = os.path.join(root,'Hilal1.png')
    imgGray = cv.imread(imgPath,cv.IMREAD_GRAYSCALE)

    orb = cv.ORB_create()
    keypoints = orb.detect(imgGray,None)
    keypoints,_ = orb.compute(imgGray,keypoints)
    imgGray = cv.drawKeypoints(imgGray,keypoints,imgGray,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    plt.figure()
    plt.imshow(imgGray)
    plt.show()

if __name__ == '__main__':
    ORB()