import cv2 as cv
img = cv.imread('soccer.jpg')  # 이미지 불러오기
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 그레이스케일 변환

grad_x = cv.Sobel(gray, cv.CV_32F, 1, 0, ksize=3)  # x축 방향 소벨 에지 검출
grad_y = cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=3)  # y축 방향 소벨 에지 검출

sobel_x = cv.convertScaleAbs(grad_x)  # x 방향 결과 정규화
sobel_y = cv.convertScaleAbs(grad_y)  # y 방향 결과 정규화

edge_strength = cv.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)  # 에지 강도 계산

cv.imshow('Original', gray)
cv.imshow('sobelx', sobel_x)
cv.imshow('sobely', sobel_y)
cv.imshow('edge strength', edge_strength)
cv.waitKey()
cv.destroyAllWindows()