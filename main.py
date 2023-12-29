from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QFont
import sys


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('App')
        self.resize(400, 400)
        self.setFont(QFont('Arial', 10, QFont.Bold))
        self.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    a = Window()
    sys.exit(app.exec_())
