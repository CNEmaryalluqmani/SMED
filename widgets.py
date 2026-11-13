"""
widgets.py - small shared Qt widgets.

ImageView: shows an image scaled to fill whatever space it is given (keeps the aspect ratio).
That is what lets the camera pictures and heat maps grow with the window / full screen.
"""
from PySide6 import QtCore, QtGui, QtWidgets


class ImageView(QtWidgets.QWidget):
    def __init__(self, min_w=160, min_h=120, bg="#000000"):
        super().__init__()
        self.pm, self.text, self.bg = None, "", QtGui.QColor(bg)
        self.setMinimumSize(min_w, min_h)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

    def setPixmap(self, pm):
        self.pm, self.text = pm, ""
        self.update()

    def setText(self, text):
        self.pm, self.text = None, text
        self.update()

    def paintEvent(self, _):
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), self.bg)
        if self.pm is not None:
            s = self.pm.scaled(self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            p.drawPixmap((self.width() - s.width()) // 2, (self.height() - s.height()) // 2, s)
        elif self.text:
            p.setPen(QtGui.QColor("#8b949e"))
            p.drawText(self.rect().adjusted(10, 10, -10, -10),
                       QtCore.Qt.AlignCenter | QtCore.Qt.TextWordWrap, self.text)