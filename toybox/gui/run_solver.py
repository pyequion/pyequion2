# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, "..")

from pyequion import EquilibriumSystem

from PyQt5.QtWidgets import QApplication
from src.main import SolverGUI

eqsys = EquilibriumSystem(['C','Ca','Na','Cl'],
                          from_elements=True,
                          activity_model="DEBYE")
app = QApplication(sys.argv)
window = SolverGUI(eqsys, "phase")
sys.exit(app.exec_())