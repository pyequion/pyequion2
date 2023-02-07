# -*- coding: utf-8 -*-
import sys

try:
    from PyQt5.QtWidgets import QApplication
except ModuleNotFoundError:
    raise ModuleNotFoundError("PyEquion2 GUI requires PyQt5")
from .main import PyEquionGUI


def run():
    app = QApplication(sys.argv)
    window = PyEquionGUI()
    sys.exit(app.exec_())
