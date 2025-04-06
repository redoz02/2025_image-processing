import skimage
import numpy as np
import cv2 as cv

orig = skimage.data.horse()  # 샘플 데이터 불러오기
img = 255 - np.uint8(orig) * 255  # 반전 처리

contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)  # 외곽선 검출

img2 = cv.cvtColor(img, cv.COLOR_GRAY2BGR)  # 컬러 이미지로 변환
cv.drawContours(img2, contours, -1, (0, 255, 0), 2)  # 외곽선 그리기
cv.imshow('Horse with Contour', img2)

contour = contours[0]  # 첫 번째 외곽선 선택

# 모멘트 계산
m = cv.moments(contour)
area = cv.contourArea(contour)
perimeter = cv.arcLength(contour, True)
roundness = (4 * np.pi * area) / (perimeter * perimeter)  # 원형도 측정
print("면적:{0}, 중심:({1},{2}), 둘레:{3}, 원형도:{4:.2f}".format(area, int(m['m10']/m['m00']), int(m['m01']/m['m00']), perimeter, roundness))

img3 = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
contour_approx = cv.approxPolyDP(contour, 8, True)  # 근사 곡선
cv.drawContours(img3, [contour_approx], -1, (0, 255, 0), 2)

hull = cv.convexHull(contour)  # 볼록 껍질 생성
cv.drawContours(img3, [hull], -1, (0, 0, 255), 2)

cv.imshow('Horse with line segments and convex hull', img3)
cv.waitKey()
cv.destroyAllWindows()