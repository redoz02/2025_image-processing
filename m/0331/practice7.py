import cv2 as cv
import numpy as np

img = cv.imread('soccer.jpg')  # 이미지 불러오기
mask = np.zeros(img.shape[:2], np.uint8)  # 마스크 초기화
bgModel = np.zeros((1, 65), np.float64)  # 배경 모델
fgModel = np.zeros((1, 65), np.float64)  # 전경 모델

BrushSiz = 9  # 브러시 크기
LColor, RColor = (255, 0, 0), (0, 255, 0)  # 왼쪽/오른쪽 색 설정

# 마우스 이벤트 콜백 함수 정의
def painting(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        cv.circle(mask, (x, y), BrushSiz, cv.GC_FGD, -1)
        cv.circle(img, (x, y), BrushSiz, LColor, -1)
    elif event == cv.EVENT_RBUTTONDOWN:
        cv.circle(mask, (x, y), BrushSiz, cv.GC_BGD, -1)
        cv.circle(img, (x, y), BrushSiz, RColor, -1)
    elif event == cv.EVENT_MOUSEMOVE and flags & cv.EVENT_FLAG_LBUTTON:
        cv.circle(mask, (x, y), BrushSiz, cv.GC_FGD, -1)
        cv.circle(img, (x, y), BrushSiz, LColor, -1)
    elif event == cv.EVENT_MOUSEMOVE and flags & cv.EVENT_FLAG_RBUTTON:
        cv.circle(mask, (x, y), BrushSiz, cv.GC_BGD, -1)
        cv.circle(img, (x, y), BrushSiz, RColor, -1)

cv.imshow('Painting', img)
cv.namedWindow('Painting')
cv.setMouseCallback('Painting', painting)  # 콜백 함수 등록

while True:
    if cv.waitKey(1) == ord('q'):  # q 누르면 루프 종료
        break

# GrabCut 수행
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)
mask2 = np.where((mask == cv.GC_BGD) | (mask == cv.GC_PR_BGD), 0, 1).astype('uint8')
cv.grabCut(img, mask, None, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_MASK)
img = img * mask2[:, :, np.newaxis]  # 마스크된 영역만 표시
cv.imshow('Grab cut image', img)
cv.waitKey()
cv.destroyAllWindows()