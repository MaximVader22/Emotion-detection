from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QCheckBox, QVBoxLayout, QLabel, QHBoxLayout
from PyQt5.QtGui import QFont, QImage
from PyQt5 import QtCore, QtGui
import sys
import cv2


faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


class StartWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.win = None
        self.setWindowTitle('Распознавание эмоций')
        self.setFixedSize(400, 600)
        self.setFont(QFont('Arial', 10, QFont.Bold))
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        start_button = QPushButton("Запустить приложение", self)
        start_button.clicked.connect(self.start_button_clicked)
        layout.addWidget(start_button)

        checkbox_video = QCheckBox("Сохранять видео")
        layout.addWidget(checkbox_video)

        choose_dir_video = QPushButton("Выбрать папку сохранения видео", self)
        layout.addWidget(choose_dir_video)

        checkbox_logs = QCheckBox("Сохранять отчёты")
        layout.addWidget(checkbox_logs)

        choose_dir_logs = QPushButton("Выбрать папку сохранения отчётов", self)
        layout.addWidget(choose_dir_logs)

        layout.insertSpacing(3, 100)
        layout.insertSpacing(1, 100)
        self.setLayout(layout)
        self.show()

    def start_button_clicked(self):
        self.win = MainWindow()
        self.win.start()
        self.win.show()
        self.close()


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.win = None
        self.camera = cv2.VideoCapture(0)
        self.label = QLabel()
        exit_button = QPushButton("Закрыть приложение", self)
        exit_button.clicked.connect(self.exit_button_clicked)
        layout = QVBoxLayout()
        self.setLayout(layout)
        layout.addWidget(self.label)
        layout.addWidget(exit_button)
        self.timer = QtCore.QTimer()
        self.timer.setInterval(50)
        self.timer.timeout.connect(self.draw_camera)

    def start(self):
        self.timer.start()

    def stop(self):
        self.timer.stop()

    def draw_camera(self):
        b, frame = self.camera.read()
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(img_gray)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = frame.shape
        bpl = 3 * width
        q_img = QImage(frame.data, width, height, bpl, QImage.Format_RGB888)
        pix = QtGui.QPixmap(q_img)
        self.label.setPixmap(pix)

    def closeEvent(self, event):
        self.stop()
        return QWidget.closeEvent(self, event)
    
    def exit_button_clicked(self):
        self.win = StartWindow()
        self.win.show()
        self.stop()
        self.close()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    a = StartWindow()
    sys.exit(app.exec_())
