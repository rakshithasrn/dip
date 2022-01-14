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

# write a program to implementation of transformation of an image

import numpy as np

import cv2 as cv

from matplotlib import pyplot as plt

#translation

img = cv.imread('btrfly.jpg',0)

rows,cols = img.shape

M = np.float32([[1,0,100],[0,1,50]])

dst = cv.warpAffine(img,M,(cols,rows))

cv.imshow('img',dst)

cv.waitKey(0)

cv.destroyAllWindows()

#rotation

img = cv.imread('btrfly.jpg',0)

rows,cols = img.shape

# cols-1 and rows-1 are the coordinate limits.

M = cv.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),90,1)

dst = cv.warpAffine(img,M,(cols,rows))

#cv.imshow('img',dst)

plt.subplot(121),plt.imshow(img),plt.title('Input')

plt.subplot(122),plt.imshow(dst),plt.title('Output')

plt.show()

cv.waitKey(0)

cv.destroyAllWindows()

![image](https://user-images.githubusercontent.com/96527199/149121385-8bc2c3ac-a189-42d4-a9a0-de35b8f5814e.png)

![image](https://user-images.githubusercontent.com/96527199/149121446-2c4390d9-f089-4d13-aedb-0b5aa84ddb54.png)


#scaling

img=cv.imread('btrfly.jpg',cv.IMREAD_COLOR)

resized=cv.resize(img,None,fx=1,fy=2,interpolation=cv.INTER_CUBIC)

#cv.imshow("original pic",img)

#cv.imshow("resized pic",resized)

plt.subplot(121),plt.imshow(img),plt.title('Input')

plt.subplot(122),plt.imshow(resized),plt.title('Output')

plt.show()

cv.waitKey()

cv.destroyAllWindows()

![image](https://user-images.githubusercontent.com/96527199/149121477-9d24e387-ef25-453c-b855-cc5397d29b75.png)

![image](https://user-images.githubusercontent.com/96527199/149121513-f271df58-91df-4413-b47f-7d58b08e7ac6.png)



#Perspective Transformation

img = cv.imread('1.jpg')

rows,cols,ch = img.shape

pts1 = np.float32([[56,65],[150,52],[28,387],[150,390]])

pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])

M = cv.getPerspectiveTransform(pts1,pts2)

dst = cv.warpPerspective(img,M,(300,300))

plt.subplot(121),plt.imshow(img),plt.title('Input')

plt.subplot(122),plt.imshow(dst),plt.title('Output')

plt.show()

![image](https://user-images.githubusercontent.com/96527199/149121558-cb510cd6-23d5-40f6-ac8d-0a3be76414c4.png)

![image](https://user-images.githubusercontent.com/96527199/149121601-4db9c874-cafa-4f4d-93eb-be6ca0ce543e.png)

# python program access image from directory

# import the modules
import os

from os import listdir
 
# get the path or directory

folder_dir = "C:\img"

for images in os.listdir(folder_dir):
 
    # check if the image end swith png or jpg or jpeg
    
    if (images.endswith(".png") or images.endswith(".jpg")\
    
        or images.endswith(".jpeg")):
        
        # display
        
        print(images)
        
        # OUTPUT:-
        
        d1.jpg
        
        d2.jpg
       
        d3.jpg
        
       # python program to find mean value of image
       
       
import cv2
  
#Save image in set directory

#Read RGB image

img = cv2.imread('d1.jpg') 

 #getting mean value
 
mean = img.mean()
  
#printing mean value

print("Mean Value for 0 channel : " + str(mean))

#Output img with window name as 'image'

cv2.imshow('image', img) 
  
#Maintain output window utill

#user presses a key

cv2.waitKey(0)        
  
#Destroying present windows on screen

cv2.destroyAllWindows() 

# output:-

Mean Value for 0 channel : 83.87289203784049

# code to merge images

from PIL import Image
  
img_01 = Image.open("d1.jpg")

img_02 = Image.open("d2.jpg")

img_01_size = img_01.size

img_02_size = img_02.size

print('img 1 size: ', img_01_size)

print('img 2 size: ', img_02_size)

new_im = Image.new('RGB', (2*img_01_size[0],2*img_01_size[1]), (250,250,250))
  
new_im.paste(img_01, (0,0))

new_im.paste(img_02, (img_01_size[0],0))

new_im.save("merged_images.png", "PNG")

new_im.show()

# output:_

img 1 size:  (259, 194)

img 2 size:  (275, 183)

# python program for sudoku

import cv2
import numpy as np

#Loading image contains lines

img = cv2.imread('sudoku.jpg')

#Convert to grayscale

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#Apply Canny edge detection, return will be a binary image

edges = cv2.Canny(gray,50,100,apertureSize = 3)

#Apply Hough Line Transform, minimum lenght of line is 200 pixels

lines = cv2.HoughLines(edges,1,np.pi/180,200)

#Print and draw line on the original image

for rho,theta in lines[0]:

 print(rho, theta)
 
 a = np.cos(theta)
 
 b = np.sin(theta)
 
 x0 = a*rho
 
 y0 = b*rho
 
 x1 = int(x0 + 1000*(-b))
 
 y1 = int(y0 + 1000*(a))
 
 x2 = int(x0 - 1000*(-b))
 
 y2 = int(y0 - 1000*(a))
 
 cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

#Show the result

cv2.imshow("Line Detection", img)

cv2.waitKey(0)

cv2.destroyAllWindows()

# OUTPUT:-

![image](https://user-images.githubusercontent.com/96527199/149463978-3185605a-bf3b-4334-940e-1cbeaf313aac.png)

27.0 1.5707964
