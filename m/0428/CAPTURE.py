from PyQt5.QtWidgets import *
import sys
import cv2 as cv

class Video(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('비디오에서 프레임 수집')  # 윈도우 이름
        self.setGeometry(200, 200, 500, 100)         # 윈도우 크기 및 위치

        # 버튼 생성
        videoButton = QPushButton('비디오 켜기', self)
        captureButton = QPushButton('프레임 잡기', self)
        saveButton = QPushButton('프레임 저장', self)
        quitButton = QPushButton('나가기', self)

        # 버튼 위치 및 크기 지정
        videoButton.setGeometry(10, 10, 100, 30)
        captureButton.setGeometry(110, 10, 100, 30)
        saveButton.setGeometry(210, 10, 100, 30)
        quitButton.setGeometry(310, 10, 100, 30)

        # 콜백 함수 연결
        videoButton.clicked.connect(self.videoFunction)
        captureButton.clicked.connect(self.captureFunction)
        saveButton.clicked.connect(self.saveFunction)
        quitButton.clicked.connect(self.quitFunction)

    def videoFunction(self):
        self.cap = cv.VideoCapture(0, cv.CAP_DSHOW)  # 카메라 연결
        if not self.cap.isOpened():
            self.close()
        while True:
            ret, self.frame = self.cap.read()
            if not ret:
                break
            cv.imshow('video display', self.frame)
            if cv.waitKey(1) == 27:  # ESC 누르면 종료
                break

    def captureFunction(self):
        self.capturedFrame = self.frame  # 프레임 저장
        cv.imshow('Captured Frame', self.capturedFrame)

    def saveFunction(self):
        fname = QFileDialog.getSaveFileName(self, '파일 저장', './')  # 파일 이름 받기
        if fname[0]:  # 파일 이름이 있으면 저장
            cv.imwrite(fname[0], self.capturedFrame)

    def quitFunction(self):
        if hasattr(self, 'cap'):
            self.cap.release()  # 카메라 해제
        cv.destroyAllWindows()
        self.close()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = Video()
    win.show()
    app.exec_()
