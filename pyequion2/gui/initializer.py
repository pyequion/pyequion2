# -*- coding: utf-8 -*-
import sys

from PyQt5.QtWidgets import (QApplication, QWidget, QLabel,
                             QTextEdit, QLineEdit,
                             QPushButton, QCheckBox,
                             QGridLayout, QVBoxLayout,
                             QHBoxLayout, QMessageBox,
                             QComboBox)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt

from .. import EquilibriumSystem
from .solver import SolverGUI


class InitializerGUI(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_ = parent #TODO: There must be some PyQt actual solution
        self.initializeUI()
    
    def initializeUI(self):
        if not self.has_parent:
            self.setGeometry(100, 100, 300, 300)
            self.setWindowTitle('PyEquion GUI Initializer')
        self.setupWidgets()
        self.show()

    def setupWidgets(self):
        main_grid = QGridLayout()
        elements_label = QLabel("Base species/elements")
        self.elements_inserter = QTextEdit()
        self.elements_inserter.setPlaceholderText(
            "Put every element in a row. Example: \n"\
            "C\n"\
            "Ca\n"\
            "Na\n"
            )
        self.from_elements_checker = QCheckBox("From elements")
        self.from_elements_checker.setToolTip("If checked, put elements above. "\
                                              "If not, put seed ions")
        self.from_elements_checker.setChecked(True)
        self.create_button = QPushButton("Create equilibrium")
        self.create_button.setToolTip("Create the equilibrium system")
        self.create_button.clicked.connect(self.create_equilibrium)
        equilibrium_type_label = QLabel("Equilibrium type:")
        self.equilibrium_type_cb = QComboBox()
        self.equilibrium_type_cb.addItems(["Aqueous equilibrium", "Phase equilibrium"])
        
        main_grid.addWidget(elements_label, 0, 0, 1, 2)
        main_grid.addWidget(self.elements_inserter, 1, 0, 1, 2)
        main_grid.addWidget(self.from_elements_checker, 2, 0, 1, 1)
        main_grid.addWidget(equilibrium_type_label, 3, 0, 1, 1)
        main_grid.addWidget(self.equilibrium_type_cb, 3, 1, 1, 1)
        main_grid.addWidget(self.create_button, 4, 0, 1, 2)
        
        self.setLayout(main_grid)
        
    def create_equilibrium(self):
        from_elements = self.from_elements_checker.isChecked()
        base_species = [s.strip() for s in self.elements_inserter.toPlainText().strip('\n').split('\n')]
        try:
            eqsys = EquilibriumSystem(base_species, from_elements=from_elements,
                                      activity_model="PITZER")
        except:
            QMessageBox.critical(self, 
                                  "Could not create equilibrium",
                                  "Could not create equilibrium. Did you set seeds correctly?",
                                  QMessageBox.Close,
                                  QMessageBox.Close)
            return
        if self.equilibrium_type_cb.currentText() == "Aqueous equilibrium":
            type_eq = "aqueous"
        elif self.equilibrium_type_cb.currentText() == "Phase equilibrium":
            type_eq = "phase"
        
        solving_gui = SolverGUI(eqsys, type_eq, self.parent_)
        self.create_new_gui(solving_gui)
    
    def create_new_gui(self, new_gui):
        if not self.has_parent:
            self.new_gui = new_gui
            self.new_gui.show()
        else:
            self.parent_.display_and_connect(self, new_gui, "Solver")

    @property
    def has_parent(self):
        return self.parent_ is not None