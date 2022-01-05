# 1. python program to perform linear transformations of an image
import cv2

import numpy as np

import matplotlib.pyplot as plt

img = cv2.imread('thumbs.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

h, w = img.shape[:2]

T = cv2.getRotationMatrix2D((w/2, h/2), 90, .5)

img = cv2.warpAffine(img, T, (w, h))

fig, ax = plt.subplots(1, figsize=(12,8))

ax.axis('off')  

plt.imshow(img)

# output:-

![image](https://user-images.githubusercontent.com/96527199/148203934-3ff41926-418b-40c3-ab5d-de4f91d0e8ad.png)


![image](https://user-images.githubusercontent.com/96527199/148198573-c0902f18-b516-4616-b6c6-a37004454ad1.png)
