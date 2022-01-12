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




# 2.Write a program simulation and display of an image,negative of an image (Binary and Gray scale)

import cv2

import matplotlib.pyplot as plt
  
# Read an image

img_bgr = cv2.imread('btrfly.jpg', 1)

plt.imshow(img_bgr)

plt.show()
  
# Histogram plotting of the image

color = ('b', 'g', 'r')
  
for i, col in enumerate(color):
      
    histr = cv2.calcHist([img_bgr], 
    
                         [i], None,
                         
                         [256], 
                         
                         [0, 256])
      
    plt.plot(histr, color = col)
      
    # Limit X - axis to 256
    
    plt.xlim([0, 256])
      
plt.show()
  
# get height and width of the image

height, width, _ = img_bgr.shape
  
for i in range(0, height - 1):

    for j in range(0, width - 1):
          
        # Get the pixel value
        
        pixel = img_bgr[i, j]
          
        # Negate each channel by 
        
        # subtracting it from 255
          
        # 1st index contains red pixel
        
        pixel[0] = 255 - pixel[0]
          
        # 2nd index contains green pixel
        
        pixel[1] = 255 - pixel[1]
          
        # 3rd index contains blue pixel
        pixel[2] = 255 - pixel[2]
          
        # Store new values in the pixel
        
        img_bgr[i, j] = pixel
  
# Display the negative transformed image

plt.imshow(img_bgr)

plt.show()
  
# Histogram plotting of the

# negative transformed image

color = ('b', 'g', 'r')
  
for i, col in enumerate(color):
      
    histr = cv2.calcHist([img_bgr],
    
                         [i], None,
                         
                         [256],
                         
                         [0, 256])
      
    plt.plot(histr, color = col)
    
    plt.xlim([0, 256])
      
plt.show()
# OUTPUT:_

![image](https://user-images.githubusercontent.com/96527199/149109865-ec5058d7-f22c-4808-8151-29bdf3f3d320.png)

![image](https://user-images.githubusercontent.com/96527199/149110181-fafb9205-2a0f-417c-88b3-cb844de424ed.png)

![image](https://user-images.githubusercontent.com/96527199/149110372-4d759aac-f478-4741-ac87-f5fdce61e0b9.png)

![image](https://user-images.githubusercontent.com/96527199/149110578-7730fd9e-2256-4527-a15f-88a28f787036.png)

# 3.Write a program to contrast stretching of a low contrast image,Histogram and histogram equalization

import cv2

# import Numpy

import numpy as np 

from matplotlib import pyplot as plt 

# reading an image using imreadmethod

my_img = cv2.imread('btrfly.jpg', 0)

equ = cv2.equalizeHist(my_img)

# stacking both the images side-by-side orientation

res = np.hstack((my_img, equ))

# showing image input vs output

cv2.imshow('image', res)

cv2.waitKey(0)

cv2.destroyAllWindows()

hist,bins = np.histogram(equ.flatten(),256,[0,256])

cdf = hist.cumsum()

cdf_normalized = cdf * float(hist.max()) / cdf.max()

plt.plot(cdf_normalized, color = 'b')

plt.hist(equ.flatten(),256,[0,256], color = 'r')

plt.xlim([0,256])

plt.legend(('cdf','histogram'), loc = 'upper left')

plt.show()

# OUTPUT:-

![image](https://user-images.githubusercontent.com/96527199/149117143-fbac6cf5-561c-4c3d-a491-4d643ecf75b1.png)


