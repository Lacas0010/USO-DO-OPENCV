import numpy as np
import cv2

img = cv2.imread("atividade_1/Gear5.png")
img = cv2.resize(img, (0, 0), None, .25, .25)
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img_gray3C = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
img_blur = cv2.blur(img,(9,9))
img_sobel = cv2.Sobel(src=img_gray, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
ret, thresh = cv2.threshold(img_gray, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img_gray, contours, -1, (0,255,0), 1)

numpy_horizontal_gray = np.hstack((img, img_gray3C))
numpy_horizontal_blur = np.hstack((img,img_blur))

cv2.imshow("Image",img)
cv2.waitKey(0)
cv2.imshow('Numpy Horizontal', numpy_horizontal_gray)
cv2.waitKey(0)
cv2.imshow('Numpy Horizontal', numpy_horizontal_blur)
cv2.waitKey(0)
cv2.imshow('Sobel X Y using Sobel() function', img_sobel)
cv2.waitKey(0)
cv2.imshow('Contours', img_gray)
cv2.waitKey(0)


