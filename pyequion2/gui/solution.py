# -*- coding: utf-8 -*-
import itertools

from PyQt5.QtWidgets import (QApplication, QWidget, QLabel,
                             QTextEdit, QLineEdit,
                             QPushButton, QCheckBox,
                             QGridLayout, QVBoxLayout,
                             QHBoxLayout, QMessageBox,
                             QComboBox, QScrollArea,
                             QFrame, QFileDialog)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt

import numpy as np


HEADER_COLOR = "cyan"


class SolutionGUI(QWidget):
    def __init__(self, solution, solver_log, type_eq, parent=None):
        super().__init__(parent)
        self.parent_ = parent #TODO: There must be some PyQt actual solution
        self.solution = solution
        self.solver_log = solver_log
        self.type_eq = type_eq
        self.initializeUI()
    
    def initializeUI(self):
        if not self.has_parent:
            self.setGeometry(100, 100, 300, 300)
            self.setWindowTitle("PyEquion GUI Solver")
        self.setupWidgets()
        self.show()
        
    def setupWidgets(self):
        self.main_layout = QGridLayout()

        save_log_button = QPushButton("Save log")
        save_log_button.clicked.connect(self.save_log_to_file)

        properties_vbox = self.make_properties_vbox()
        properties_vbox.setContentsMargins(5, 5, 25, 5)
        
        species_grid = self.make_species_grid()
        species_grid.setContentsMargins(25, 5, 25, 5)
        
        if self.type_eq == "aqueous":
            phases_grid = self.make_saturation_indexes()
        elif self.type_eq == "phase":
            phases_grid = self.make_phase_molals()
        phases_grid.setContentsMargins(25, 5, 5, 5)

        self.main_layout.addLayout(properties_vbox, 0, 0)
        self.main_layout.addLayout(species_grid, 0, 1)
        self.main_layout.addLayout(phases_grid, 0, 2)
        self.main_layout.addWidget(save_log_button, 1, 0)
        
        self.setLayout(self.main_layout)
        
    def make_species_grid(self):
        sorted_species = sorted(self.solution.molals,
                                   key=lambda k : self.solution.molals[k],
                                   reverse=True)
        species_grid = QGridLayout()
        
        title_species_label = QLabel("Component")
        title_molals_label = QLabel("Molal")
        title_act_label = QLabel("Activity")
        title_fraction_label = QLabel("Mole fraction")
        
        species_grid.addWidget(title_species_label, 0, 0)
        species_grid.addWidget(title_molals_label, 0, 1)
        species_grid.addWidget(title_act_label, 0, 2)
        species_grid.addWidget(title_fraction_label, 0, 3)
        
        for i, specie in enumerate(sorted_species, 1):
            # specie_hbox = QHBoxLayout()
            specie_label = QLabel(specie)
            molals_label = self.show_value_label(specie, self.solution.molals)
            # conc_label = self.show_value_label(specie, self.solution.concentrations)
            act_label = self.show_value_label(specie, self.solution.activities)
            fraction_label = self.show_value_label(specie, self.solution.mole_fractions)
            species_grid.addWidget(specie_label, i, 0)
            species_grid.addWidget(molals_label, i, 1)
            species_grid.addWidget(act_label, i, 2)
            species_grid.addWidget(fraction_label, i, 3)

        sorted_elements = sorted(self.solution.elements_molals,
                                   key=lambda k : self.solution.elements_molals[k],
                                   reverse=True)
        for j, element in enumerate(sorted_elements, i+1):
            specie_label = QLabel(element)
            molals_label = self.show_value_label(element, self.solution.elements_molals)
            act_label = QLabel("")
            fraction_label = QLabel("")
            species_grid.addWidget(specie_label, j, 0)
            species_grid.addWidget(molals_label, j, 1)
            species_grid.addWidget(act_label, j, 2)
            species_grid.addWidget(fraction_label, j, 3)

        species_grid.setSpacing(0)
        items = (species_grid.itemAt(i) for i in range(species_grid.count())) 
        for item in items:
            item.widget().setStyleSheet("border: 1px solid black;")
            
        for i in range(species_grid.columnCount()):
            species_grid.setColumnStretch(i, 1)
        return species_grid
        
    def make_saturation_indexes(self):
        phases = sorted(self.solution.saturation_indexes,
                        key = lambda k : self.solution.saturation_indexes[k],
                        reverse=True)
        phases_grid = QGridLayout()
        
        title_phase = QLabel("Phase")
        title_si = QLabel("SI")
        title_satur = QLabel("Saturation")
        phases_grid.addWidget(title_phase, 0, 0)
        phases_grid.addWidget(title_si, 0, 1)
        phases_grid.addWidget(title_satur, 0, 2)
        
        for i, phase in enumerate(phases, 1):
            phase_label = QLabel(phase)
            si_label = self.show_value_label(phase, self.solution.saturation_indexes)
            satur_label = self.show_value_label(phase, self.solution.saturations)
            phases_grid.addWidget(phase_label, i, 0)
            phases_grid.addWidget(si_label, i, 1)
            phases_grid.addWidget(satur_label, i, 2)

        phases_grid.setSpacing(0)
        items = (phases_grid.itemAt(i) for i in range(phases_grid.count())) 
        for item in items:
            item.widget().setStyleSheet("border: 1px solid black;")

        for i in range(phases_grid.columnCount()):
            phases_grid.setColumnStretch(i, 1)

        return phases_grid

    def make_phase_molals(self):
        phases_grid = QGridLayout()
        solid_phases = sorted(self.solution.solid_molals,
                        key = lambda k : self.solution.solid_molals[k],
                        reverse=True)
        gas_phases = sorted(self.solution.gas_molals,
                        key = lambda k : self.solution.gas_molals[k],
                        reverse=True)
        
        title_phase = QLabel("Phase")
        title_molal = QLabel("Molals")
        phases_grid.addWidget(title_phase, 0, 0)
        phases_grid.addWidget(title_molal, 0, 1)
        
        i = 0
        for i, solid_phase in enumerate(solid_phases, 1):
            phase_label = QLabel(solid_phase)
            molal_label = self.show_value_label(solid_phase,
                                                self.solution.solid_molals)
            phases_grid.addWidget(phase_label, i, 0)
            phases_grid.addWidget(molal_label, i, 1)
        
        for j, gas_phase in enumerate(gas_phases, i+1):
            phase_label = QLabel(gas_phase)
            molal_label = self.show_value_label(gas_phase, self.solution.gas_molals)
            phases_grid.addWidget(phase_label, j, 0)
            phases_grid.addWidget(molal_label, j, 1)

        phases_grid.setSpacing(0)
        items = (phases_grid.itemAt(i) for i in range(phases_grid.count())) 
        for item in items:
            item.widget().setStyleSheet("border: 1px solid black;")

        for i in range(phases_grid.columnCount()):
            phases_grid.setColumnStretch(i, 1)

        return phases_grid

    def make_properties_vbox(self):
        properties_vbox = QVBoxLayout()
        
        ph_str = "pH = {:.2f}".format(self.solution.ph)
        ionic_strength_str = "I = {:.2f} molals".format(self.solution.ionic_strength)
        conductivity_str = "\u03C3 = {:.2f} S/m".format(self.solution.electrical_conductivity)
        balance_str = self.solver_log
        type_equilibrium_marker = "aqueous only" if (self.type_eq == "aqueous") else "phases precipitation"
        type_equilibrium_string = "Equilibrium type: {0}".format(type_equilibrium_marker)
        
        properties_vbox.addWidget(QLabel("Properties:"))
        properties_vbox.addWidget(QLabel(ph_str))
        properties_vbox.addWidget(QLabel(ionic_strength_str))
        properties_vbox.addWidget(QLabel(conductivity_str))
        properties_vbox.addWidget(QLabel(" "))
        properties_vbox.addWidget(QLabel("Balance conditions"))
        properties_vbox.addWidget(QLabel(balance_str))
        properties_vbox.addWidget(QLabel(type_equilibrium_string))
        properties_vbox.addStretch()

        return properties_vbox

    def show_value_label(self, val, d):
        if val in d:
            label_str = "{:.2e}".format(d[val])
        else:
            label_str = ""
        label = QLabel(label_str)
        return label
    
    def save_log_to_file(self):
        #else
        file_name, _ = QFileDialog.getSaveFileName(self, 'Save File',
            "","Text File (*.txt)")
        try:
            self.solution.savelog(file_name)
        except:
            QMessageBox.information(self, "Error", 
                "Unable to save file.", QMessageBox.Ok)
    
    @property
    def has_parent(self):
        return self.parent_ is not None