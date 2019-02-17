import cv2
import numpy as np
#import image
image = cv2.imread('/home/swetha/7thsem/ocr-kannada/images/one.jpg')
cv2.imshow('orig',image)
cv2.waitKey(0)

#grayscale
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow('gray',gray)
cv2.waitKey(0)

#binary
ret,thresh = cv2.threshold(gray,128,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
cv2.imshow('second',thresh)
cv2.waitKey(0)

#dilation
kernel = np.ones((5,100), np.uint8)
img_dilation = cv2.dilate(thresh, kernel, iterations=1)
cv2.imshow('dilated',img_dilation)
cv2.waitKey(0)
cv2.imwrite("dil1.jpg",img_dilation)

#find contours
im2,ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#print(ctrs)
#cv2.drawContours(image, ctrs,-1,(0,255,0),3)
#cv2.imshow("op", image)
#cv2.waitKey(0)
widths=[]
#sort contours
#sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

for i, ctr in enumerate(ctrs):
    # Get bounding box
    x, y, w, h = cv2.boundingRect(ctr)
    
    # Getting ROI
    roi = image[y:y+h, x:x+w]

    # show ROI
    
    #cv2.imshow('segment no:'+str(i),roi)
    if(h>50 and w>300):
    	cv2.rectangle(image,(x,y),( x + w, y + h ),(90,0,255),2)
    #cv2.waitKey(0)

cv2.imshow('marked areas',image)
cv2.waitKey(0)
cv2.imwrite("op1-5.jpg",image)



