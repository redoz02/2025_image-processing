import skimage
import numpy as np
import cv2 as cv

img = skimage.data.coffee()  # 샘플 영상 불러오기
sp_img1 = skimage.segmentation.slic(img, compactness=20.0, n_segments=600)  # SLIC 알고리즘 (compactness=20)
sp_img2 = skimage.segmentation.slic(img, compactness=40.0, n_segments=600)  # SLIC 알고리즘 (compactness=40)

slic1 = skimage.segmentation.mark_boundaries(img, sp_img1)  # 경계 표시
slic2 = skimage.segmentation.mark_boundaries(img, sp_img2)

slic1 = np.uint8(slic1 * 255.0)  # 정수형 변환
slic2 = np.uint8(slic2 * 255.0)

cv.imshow('Super pixels (compact 20)', slic1)
cv.imshow('Super pixels (compact 40)', slic2)
cv.waitKey()
cv.destroyAllWindows()