# -*- coding: utf-8 -*-
import sys

from PyQt5.QtWidgets import QApplication

from .main import PyEquionGUI


def run():
    app = QApplication(sys.argv)
    window = PyEquionGUI()
    sys.exit(app.exec_())
