import cv2 as cv
import numpy as np
from PyQt5.QtWidgets import *
import sys
import winsound

class TrafficWeak(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('교통약자 보호')
        self.setGeometry(200, 200, 700, 200)

        # 버튼 생성
        signButton = QPushButton('표지판 등록', self)
        roadButton = QPushButton('도로 영상 불러옴', self)
        recognitionButton = QPushButton('인식', self)
        quitButton = QPushButton('나가기', self)
        self.label = QLabel('환영합니다!', self)

        # 버튼 위치
        signButton.setGeometry(10, 10, 100, 30)
        roadButton.setGeometry(110, 10, 100, 30)
        recognitionButton.setGeometry(210, 10, 100, 30)
        quitButton.setGeometry(510, 10, 100, 30)
        self.label.setGeometry(10, 40, 600, 170)

        # 버튼 이벤트 연결
        signButton.clicked.connect(self.signFunction)
        roadButton.clicked.connect(self.roadFunction)
        recognitionButton.clicked.connect(self.recognitionFunction)
        quitButton.clicked.connect(self.quitFunction)

        # 표지판 이미지 목록과 대응 텍스트
        self.signFiles = [
            ['child.png', '어린이'],
            ['elder.png', '노인'],
            ['disabled.png', '장애인']
        ]
        self.signImgs = []  # 로딩된 표지판 이미지 저장 리스트

    def signFunction(self):
        self.label.clear()
        self.label.setText('교통약자 표지판을 등록합니다.')

        for fname, _ in self.signFiles:
            self.signImgs.append(cv.imread(fname))
            cv.imshow(fname, self.signImgs[-1])

    def roadFunction(self):
        # 표지판 먼저 등록되었는지 확인
        if self.signImgs == []:
            self.label.setText("먼저 표지판을 등록하세요.")
        else:
            fname = QFileDialog.getOpenFileName(self, '파일 읽기', './')
            self.roadImg = cv.imread(fname[0])
            if self.roadImg is None:
                sys.exit('파일을 찾을 수 없습니다.')

            cv.imshow('Road scene', self.roadImg)

    def recognitionFunction(self):
        if self.roadImg is None:
            self.label.setText("먼저 도로 영상을 입력하세요.")
        else:
            # SIFT 객체 생성
            sift = cv.SIFT_create()
            KD = []  # 키포인트와 기술자 쌍 저장

            for img in self.signImgs:
                gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                KD.append(sift.detectAndCompute(gray, None))

            grayRoad = cv.cvtColor(self.roadImg, cv.COLOR_BGR2GRAY)
            road_kp, road_des = sift.detectAndCompute(grayRoad, None)

            # 매칭 객체 생성 (FLANN 방식)
            matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
            GM = []  # Good Match 저장

            for sign_kp, sign_des in KD:
                knn_match = matcher.knnMatch(sign_des, road_des, 2)
                T = 0.7
                good_match = []

                # 좋은 매칭 필터링 (비율 테스트)
                for nearest1, nearest2 in knn_match:
                    if nearest1.distance / nearest2.distance < T:
                        good_match.append(nearest1)
                GM.append(good_match)

            # 매칭 수가 가장 많은 표지판 인덱스 찾기
            best = GM.index(max(GM, key=len))

            # 인식 결과 출력 및 경고음
            img_match = self.roadImg  # 매칭 결과 이미지 생략 (시각화 추가 가능)
            cv.imshow('Matches and Homography', img_match)
            self.label.setText(self.signFiles[best][1] + ' 보호구역입니다. 30km로 서행하세요.')
            winsound.Beep(3000, 500)

    def quitFunction(self):
        cv.destroyAllWindows()
        self.close()

# 실행 코드
app = QApplication(sys.argv)
win = TrafficWeak()
win.show()
app.exec_()
