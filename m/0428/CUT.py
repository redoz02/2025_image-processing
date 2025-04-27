import cv2 as cv
import numpy as np
import sys
from PyQt5.QtWidgets import *

class Orim(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('오림')  # 윈도우 타이틀
        self.setGeometry(200, 200, 700, 200)  # 창 위치와 크기

        # 버튼 생성
        fileButton = QPushButton('파일', self)
        paintButton = QPushButton('페인팅', self)
        cutButton = QPushButton('오림', self)
        incButton = QPushButton('+', self)
        decButton = QPushButton('-', self)
        saveButton = QPushButton('저장', self)
        quitButton = QPushButton('나가기', self)

        # 버튼 위치 지정
        fileButton.setGeometry(10, 10, 100, 30)
        paintButton.setGeometry(110, 10, 100, 30)
        cutButton.setGeometry(210, 10, 100, 30)
        incButton.setGeometry(310, 10, 50, 30)
        decButton.setGeometry(360, 10, 50, 30)
        saveButton.setGeometry(410, 10, 100, 30)
        quitButton.setGeometry(510, 10, 100, 30)

        # 버튼 클릭 이벤트와 함수 연결
        fileButton.clicked.connect(self.fileOpenFunction)
        paintButton.clicked.connect(self.paintFunction)
        cutButton.clicked.connect(self.cutFunction)
        incButton.clicked.connect(self.incFunction)
        decButton.clicked.connect(self.decFunction)
        saveButton.clicked.connect(self.saveFunction)
        quitButton.clicked.connect(self.quitFunction)

        # 초기 설정값
        self.BrushSiz = 5  # 브러시 크기
        self.LColor, self.RColor = (255, 0, 0), (0, 0, 255)  # 파란색(물체), 빨간색(배경)

    def fileOpenFunction(self):
        # 이미지 파일 열기
        fname = QFileDialog.getOpenFileName(self, 'Open file', './')
        self.img = cv.imread(fname[0])
        if self.img is None:
            sys.exit('파일을 찾을 수 없습니다.')

        self.img_show = np.copy(self.img)  # 표시용 이미지 복사
        cv.imshow('Painting', self.img_show)

        # GrabCut 마스크 초기화 (전부 배경으로 설정)
        self.mask = np.zeros((self.img.shape[0], self.img.shape[1]), np.uint8)
        self.mask[:, :] = cv.GC_PR_BGD

    def paintFunction(self):
        # 마우스 콜백 연결
        cv.setMouseCallback('Painting', self.painting)

    def painting(self, event, x, y, flags, param):
        # 왼쪽 클릭 시 파란색(물체)
        if event == cv.EVENT_LBUTTONDOWN:
            cv.circle(self.img_show, (x, y), self.BrushSiz, self.LColor, -1)
            cv.circle(self.mask, (x, y), self.BrushSiz, cv.GC_FGD, -1)

        # 오른쪽 클릭 시 빨간색(배경)
        elif event == cv.EVENT_RBUTTONDOWN:
            cv.circle(self.img_show, (x, y), self.BrushSiz, self.RColor, -1)
            cv.circle(self.mask, (x, y), self.BrushSiz, cv.GC_BGD, -1)

        # 마우스 이동 중 (드래그)
        elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_LBUTTON:
            cv.circle(self.img_show, (x, y), self.BrushSiz, self.LColor, -1)
            cv.circle(self.mask, (x, y), self.BrushSiz, cv.GC_FGD, -1)

        elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_RBUTTON:
            cv.circle(self.img_show, (x, y), self.BrushSiz, self.RColor, -1)
            cv.circle(self.mask, (x, y), self.BrushSiz, cv.GC_BGD, -1)

        # 갱신된 이미지 보여주기
        cv.imshow('Painting', self.img_show)

    def cutFunction(self):
        # GrabCut 실행
        background = np.zeros((1, 65), np.float64)
        foreground = np.zeros((1, 65), np.float64)
        cv.grabCut(self.img, self.mask, None, background, foreground, 5, cv.GC_INIT_WITH_MASK)

        # GrabCut 결과에서 전경 추출
        mask2 = np.where((self.mask == 2) | (self.mask == 0), 0, 1).astype('uint8')
        self.grabImg = self.img * mask2[:, :, np.newaxis]  # 전경만 남김
        cv.imshow('Scissoring', self.grabImg)

    def incFunction(self):
        self.BrushSiz = min(20, self.BrushSiz + 1)  # 브러시 최대 크기 제한

    def decFunction(self):
        self.BrushSiz = max(1, self.BrushSiz - 1)  # 브러시 최소 크기 제한

    def saveFunction(self):
        # 결과 이미지 저장
        fname = QFileDialog.getSaveFileName(self, '파일 저장', './')
        cv.imwrite(fname[0], self.grabImg)

    def quitFunction(self):
        # 창 닫기
        cv.destroyAllWindows()
        self.close()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = Orim()
    win.show()
    app.exec_()
