from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QCheckBox, QVBoxLayout, QLabel, QHBoxLayout, QSlider
from PyQt5.QtGui import QFont, QImage
from PyQt5 import QtCore, QtGui
from tensorflow.keras.models import load_model
from sys import argv, exit
from json import load as json_load, dump as json_dump
import numpy as np
import cv2

# Модель по распознаванию эмоций
model = load_model("emotion_recognition.h5")

# SSD обнаружение лиц от OpenCV
proto_txt_path = "proto.txt"
model_path = "detection_model.caffemodel"
detection_model = cv2.dnn.readNetFromCaffe(proto_txt_path, model_path)

# Каждой эмоции будет соответствовать свой цвет
emotions = ("злость", "страх", "счастье", "спокойствие", "грусть", "удивление")
emotion_colors = ((255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255))


# функция для обработки кадра веб камеры
def process_the_frame(frame, sensitivity, gray=False):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    detection_model.setInput(blob)
    output = np.squeeze(detection_model.forward())

    if gray:
        output_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        output_frame = cv2.cvtColor(output_frame, cv2.COLOR_GRAY2RGB)
    else:
        output_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    for i in range(0, output.shape[0]):
        confidence = output[i, 2]
        if confidence * 100 >= sensitivity:
            box = output[i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            face = frame[y1:y2, x1:x2]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face, (64, 64))
            face = np.array([[elem for elem in elem2] for elem2 in face])
            face = np.expand_dims(face, axis=0)
            prediction = model.predict(face, verbose=0)
            cv2.rectangle(output_frame, (x1, y1), (x2, y2),
                          emotion_colors[max(enumerate(prediction[0]), key=lambda el: el[1])[0]], 2)

    height, width, channel = frame.shape
    bpl = 3 * width
    q_img = QImage(output_frame.data, width, height, bpl, QImage.Format_RGB888)
    return QtGui.QPixmap(q_img)


# Начальное окно с возможностью настройки сохранения видео
class StartWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.win = None
        self.setWindowTitle('Распознавание эмоций')
        self.setFixedSize(400, 400)
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

        layout.insertSpacing(1, 100)
        self.setLayout(layout)
        self.show()

    def start_button_clicked(self):
        self.win = AppSetting()
        self.close()


# Окно для настройки выходного изображения
class AppSetting(QWidget):
    def __init__(self):
        super().__init__()
        self.win = None
        self.camera = cv2.VideoCapture(0)
        self.setWindowTitle('Распознавание эмоций')

        with open('config.json') as file:
            self.params = json_load(file)
        self.set_camera_params()

        self.output_video = QLabel()
        self.init_ui()
        self.move(400, 50)

        self.timer = QtCore.QTimer()
        self.timer.setInterval(50)
        self.timer.timeout.connect(self.draw_camera)
        self.timer.start()

    def init_ui(self):
        layout = QVBoxLayout()

        title = QLabel("Настройте яркость экрана как в примерах")
        title.setFont(QFont('Arial', 20, QFont.Bold))
        layout.addWidget(title)

        setting = QHBoxLayout()
        examples = QHBoxLayout()

        label = QLabel(self)
        pixmap = QtGui.QPixmap('examples/1.png')
        pixmap = pixmap.scaled(256, 256, transformMode=QtCore.Qt.SmoothTransformation)
        label.setPixmap(pixmap)
        examples.addWidget(label)

        label = QLabel(self)
        pixmap = QtGui.QPixmap('examples/2.png')
        pixmap = pixmap.scaled(256, 256, transformMode=QtCore.Qt.SmoothTransformation)
        label.setPixmap(pixmap)
        examples.addWidget(label)

        label = QLabel(self)
        pixmap = QtGui.QPixmap('examples/3.png')
        pixmap = pixmap.scaled(256, 256, transformMode=QtCore.Qt.SmoothTransformation)
        label.setPixmap(pixmap)
        examples.addWidget(label)

        left_side = QVBoxLayout()
        left_side.addLayout(examples)
        left_side.addWidget(self.output_video)
        setting.addLayout(left_side)

        v_box = QVBoxLayout()
        title = QLabel("Яркость")
        brightness_slider = QSlider(QtCore.Qt.Orientation.Vertical, self)
        brightness_slider.setRange(0, 100)
        brightness_slider.setValue(self.params["brightness"])
        v_box.addWidget(title)
        v_box.addWidget(brightness_slider)
        brightness_slider.valueChanged.connect(self.brightness_changed)
        setting.addLayout(v_box)

        v_box = QVBoxLayout()
        title = QLabel("Контраст")
        contrast_slider = QSlider(QtCore.Qt.Orientation.Vertical, self)
        contrast_slider.setRange(0, 100)
        contrast_slider.setValue(self.params["contrast"])
        v_box.addWidget(title)
        v_box.addWidget(contrast_slider)
        contrast_slider.valueChanged.connect(self.contrast_changed)
        setting.addLayout(v_box)

        v_box = QVBoxLayout()
        title = QLabel("Насыщенность")
        saturation_slider = QSlider(QtCore.Qt.Orientation.Vertical, self)
        saturation_slider.setRange(0, 100)
        saturation_slider.setValue(self.params["saturation"])
        v_box.addWidget(title)
        v_box.addWidget(saturation_slider)
        saturation_slider.valueChanged.connect(self.saturation_changed)
        setting.addLayout(v_box)

        v_box = QVBoxLayout()
        title = QLabel("Чувствительность\nраспознавания\nлиц")
        sensitivity_slider = QSlider(QtCore.Qt.Orientation.Vertical, self)
        sensitivity_slider.setRange(0, 100)
        sensitivity_slider.setValue(self.params["sensitivity"])
        v_box.addWidget(title)
        v_box.addWidget(sensitivity_slider)
        sensitivity_slider.valueChanged.connect(self.sensitivity_changed)
        setting.addLayout(v_box)

        layout.addLayout(setting)

        reset_button = QPushButton("Сбросить настройки")
        reset_button.clicked.connect(self.reset_button_clicked)
        layout.addWidget(reset_button)

        end_setting_button = QPushButton("Завершить настройку")
        end_setting_button.clicked.connect(self.end_setting_button_clicked)
        layout.addWidget(end_setting_button)

        return_button = QPushButton("Вернуться к начальному экрану")
        return_button.clicked.connect(self.return_button_clicked)
        layout.addWidget(return_button)

        self.setLayout(layout)

        self.show()

    def draw_camera(self):
        b, frame = self.camera.read()
        image = process_the_frame(frame, self.params['sensitivity'], gray=True)
        self.output_video.setPixmap(image)

    def closeEvent(self, event):
        self.timer.stop()
        return QWidget.closeEvent(self, event)

    def reset_button_clicked(self):
        params = {"brightness": 0, "contrast": 0, "saturation": 64, "sensitivity": 50}
        with open('config.json', 'w') as f:
            json_dump(params, f)
        self.set_camera_params()
        self.win = AppSetting()
        self.close()

    def end_setting_button_clicked(self):
        self.win = MainWindow()
        self.close()

    def return_button_clicked(self):
        self.win = StartWindow()
        self.close()

    def set_camera_params(self):
        self.camera.set(10, self.params["brightness"])
        self.camera.set(11, self.params["contrast"])
        self.camera.set(12, self.params["saturation"])

    def brightness_changed(self, event):
        self.params['brightness'] = event
        with open('config.json', 'w') as f:
            json_dump(self.params, f)
        self.set_camera_params()

    def contrast_changed(self, event):
        self.params['contrast'] = event
        with open('config.json', 'w') as f:
            json_dump(self.params, f)
        self.set_camera_params()

    def saturation_changed(self, event):
        self.params['saturation'] = event
        with open('config.json', 'w') as f:
            json_dump(self.params, f)
        self.set_camera_params()

    def sensitivity_changed(self, event):
        self.params['sensitivity'] = event
        with open('config.json', 'w') as f:
            json_dump(self.params, f)
        self.set_camera_params()


# Основное окно, распознавание эмоций
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.win = None
        self.camera = cv2.VideoCapture(0)
        self.setWindowTitle('Распознавание эмоций')
        self.output_video = QLabel()
        self.init_ui()

        with open('config.json') as file:
            self.params = json_load(file)
        self.camera.set(10, self.params["brightness"])
        self.camera.set(11, self.params["contrast"])
        self.camera.set(12, self.params["saturation"])

        self.timer = QtCore.QTimer()
        self.timer.setInterval(50)
        self.timer.timeout.connect(self.draw_camera)
        self.timer.start()

    def init_ui(self):
        exit_button = QPushButton("Закрыть приложение", self)
        exit_button.clicked.connect(self.exit_button_clicked)

        return_button = QPushButton("Вернуться к настройке", self)
        return_button.clicked.connect(self.return_button_clicked)

        layout = QVBoxLayout()
        self.setLayout(layout)
        layout.addWidget(self.output_video)
        layout.addWidget(return_button)
        layout.addWidget(exit_button)

        self.show()

    def draw_camera(self):
        b, frame = self.camera.read()
        image = process_the_frame(frame, self.params['sensitivity'])
        self.output_video.setPixmap(image)

    def closeEvent(self, event):
        self.timer.stop()
        return QWidget.closeEvent(self, event)
    
    def exit_button_clicked(self):
        self.win = StartWindow()
        self.win.show()
        self.close()

    def return_button_clicked(self):
        self.win = AppSetting()
        self.win.show()
        self.close()


if __name__ == '__main__':
    app = QApplication(argv)
    a = StartWindow()
    exit(app.exec_())
