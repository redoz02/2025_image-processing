import cv2 as cv
img = cv.imread('apples.jpg')  # 이미지 불러오기
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 200, param1=150, param2=20, minRadius=50, maxRadius=120)  # 허프 원 검출

for i in circles[0]:
    cv.circle(img, (int(i[0]), int(i[1])), int(i[2]), (255, 0, 0), 2)  # 원 그리기

cv.imshow('apple detection', img)
cv.waitKey()
cv.destroyAllWindows()