import cv2
import numpy as np
import matplotlib.pyplot as plt
im_gray = cv2.imread('/home/swetha/7thsem/ocr-kannada/images/one.jpg')
(thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#cv2.imwrite('bw_image.png', im_bw)
kernel = np.ones((2,2), np.uint8)
img_erosion = cv2.erode(im_bw, kernel, iterations=1)
img_dilation = cv2.dilate(img_erosion, kernel, iterations=1)
 
cv2.imshow('Input', im_bw)
'''
#cv2.imshow('Erosion', img_erosion)
#cv2.imshow('Dilation', img_dilation)
#x_sum = cv2.reduce(img_dilation, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32S) 
y_sum = cv2.reduce(img_dilation, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32S) 
#print(len(x_sum[0]))
horproj=[]
(rows,cols)=img_dilation.shape
for i in range(rows):
	horproj.append(y_sum[i][0])
print(len(horproj))
plt.plot(horproj)
plt.show()
print(rows,cols)
'''
cv2.waitKey(0)
