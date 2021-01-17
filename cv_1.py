# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 18:35:48 2021

@author: SOUVIK ROY
"""
import os
print(os.getcwd())
os.chdir('G:/mtech/2nd Semester/Computer Vision/practice')
print(os.getcwd())
import cv2
print(cv2.__version__)
from PIL import Image
Img=Image.open('prac_picture_1.png') # Open Image
import matplotlib.pyplot as plt
plt.imshow(Img) #Display Image
import numpy as np
print(np.shape(Img)) # see the size(1960, 1960, 3)=(a,b,c) where a=no of rows,b=no of columns, c= RGB
img1=np.asarray(Img)#  convert into numpy array
plt.imshow(img1[:,:,1],cmap='gray')#read only one channel
#see pixel values 
#0 refers to R channel 
#1 refers to G channel 
#2 refers to B channel 
img1[1:10,1:10,1]
# see each channel of RGB color image
r,g,b=Img.split()
img1=np.asarray(g)
plt.imshow(Img)
plt.imshow(img1)
from skimage.color import rgb2hsv
hsvimg=rgb2hsv(Img)
print(img1[1:10,1:10])
print(hsvimg[1:10,1:10,0])
print(hsvimg[1:10,1:10,1])
print(hsvimg[1:10,1:10,2])
#0 refers to H channel 
#1 refers to S channel 
#2 refers to V channel 
#if RGB=255 then H=S=0 and V=1
img=cv2.imread("prac_picture_1.png")#Read Image
#Plot a histogram
histogram_image=cv2.calcHist([img],[0],None,[256],[0,256])
hist,bins=np.histogram(img.ravel(),256,[0,256])
np.shape(hist)#(256,)
hist[1:10]##array([220625, 146202,  85329,  61733,  60020,  58527,  56293,  57878,68660], dtype=int64)
##flaten the histogram
plt.title('Histogram')
plt.xlabel('Intensity')
plt.ylabel('Number of Pixels')
plt.hist(img.ravel(),256,[0,256])
#view color channels
color=['b','g','r']
#seperate the colors and plot the histogram
for i,col in enumerate(color):
    hist=cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(hist,color=col)
    plt.xlim([0,256])
    plt.title('Histogram for RGB ')
    plt.xlabel('Intensity')
    plt.ylabel('Number of Pixels')
plt.show()
# mean of image on histogram
np.mean(img1)
# median of image on histogram
np.median(img1)
# mode of image on histogram
from scipy import stats
stats.mode(img1)
# standard deviation of image on histogram
np.std(img1)
# variance of image on histogram
np.var(img1)
# count of image on histogram
np.sum(img1)
# Minimum of image on histogram
np.min(img1)
# Maximum of image on histogram
np.max(img1)
