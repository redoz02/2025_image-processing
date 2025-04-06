import cv2 as cv
img = cv.imread('soccer.jpg')  # 이미지 불러오기
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 그레이스케일 변환

canny1 = cv.Canny(gray, 50, 150)  # 캐니 에지 검출 (T1=50, T2=150)
canny2 = cv.Canny(gray, 100, 200)  # 캐니 에지 검출 (T1=100, T2=200)

cv.imshow('Original', gray)
cv.imshow('Canny1', canny1)
cv.imshow('Canny2', canny2)
cv.waitKey()
cv.destroyAllWindows()