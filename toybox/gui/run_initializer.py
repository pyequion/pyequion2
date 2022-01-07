# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, "..")

from PyQt5.QtWidgets import QApplication
from src.main import InitializerGUI

app = QApplication(sys.argv)
window = InitializerGUI()
sys.exit(app.exec_())