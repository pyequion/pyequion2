# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, "..")

from pyequion import EquilibriumSystem

from PyQt5.QtWidgets import QApplication
from src.main import SolutionGUI

eqsys = EquilibriumSystem(['C','Ca','Na','Cl'],
                          from_elements=True,
                          activity_model="DEBYE")

elements_balance = {'Ca':0.028, 'C':0.065, 'Na':0.075, 'Cl':0.056}
TK = 273.15 + 95.0
solution, res = eqsys.solve_equilibrium_elements_balance_phases(TK, elements_balance, tol=1e-12)

app = QApplication(sys.argv)
window = SolutionGUI(solution, "phase")
sys.exit(app.exec_())