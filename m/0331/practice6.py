import skimage
import numpy as np
import cv2 as cv
import time

coffee = skimage.data.coffee()
start = time.time()
slic = skimage.segmentation.slic(coffee, compactness=20.0, n_segments=600, start_label=1)  # SLIC 수행
g = skimage.future.graph.rag_mean_color(coffee, slic, mode='similarity')
ncut = skimage.future.graph.cut_normalized(slic, g)  # 정규화 컷 실행
print(coffee.shape, 'coffee 영상을 분할하는데', time.time() - start, '초 소요')

marking = skimage.segmentation.mark_boundaries(coffee, ncut)
marking = np.uint8(marking * 255.0)

cv.imshow('Normalized cut', cv.cvtColor(marking, cv.COLOR_RGB2BGR))
cv.waitKey()
cv.destroyAllWindows()
