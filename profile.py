import cv2
import numpy as np
#import image
image = cv2.imread('/home/swetha/7thsem/ocr-kannada/images/6.jpg')

image=cv2.fastNlMeansDenoisingColored(image,None,10,10,7,21)
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
kernel = np.ones((3,50), np.uint8)
img_dilation = cv2.dilate(thresh, kernel, iterations=1)


line=[]
horproj=[]
prev=0
(rows,cols)=img_dilation.shape
for i in range(rows):
	x=0
	for j in range(cols):
		if(img_dilation[i][j]==255):
			x=x+1
	if(x>50):
		if(prev==0):
			line.append(i)
			#cv2.line(img_dilation,(i,0),(i,cols),(255,0,0),5)
			prev=1
	else:
		prev=0
	horproj.append(x)
for i in line:
	cv2.line(image,(0,i),(cols,i),(255,0,0),5)
cv2.imwrite('op.jpg',image)
#cv2.waitKey(0)
print(line)
print(sum(horproj)//rows)
