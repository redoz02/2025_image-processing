import cv2 as cv
import numpy as np
img = cv.imread('soccer.jpg')  # 이미지 불러오기
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
canny = cv.Canny(gray, 100, 200)  # 캐니 에지 검출

contour, _ = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)  # 외곽선 검출

for i in range(len(contour)):
    if contour[i].shape[0] > 100:  # 100개 이상 점을 가진 외곽선만 사용
        cv.drawContours(img, contour, i, (0, 255, 0), 3)  # 초록색 외곽선 그리기

cv.imshow('Original with contours', img)
cv.imshow('Canny', canny)
cv.waitKey()
cv.destroyAllWindows()