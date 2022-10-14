import cv2 
import numpy as np

img = cv2.imread("input/englishimg4.png")
imgresize = cv2.imread("input/img3resize.jpg")
imgresize = cv2.resize(imgresize, (600, 800))
pts_src = np.array([(467, 31), (513, 596), (96, 759), (23, 139)])
# img = cv2.resize(img, (600, 800)) 
pts_dts = np.array([(23, 180), (447, 118), (569, 673), (104, 772)])
h, status = cv2.findHomography(pts_src, pts_dts)
dst_img = cv2.warpPerspective(imgresize, h, (600, 800))
print(h)
print(imgresize.shape)


cv2.imshow("Image", img)
cv2.waitKey(0)