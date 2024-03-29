# -*- coding: utf-8 -*-
import itertools
import collections

from PyQt5.QtWidgets import (QApplication, QWidget, QLabel,
                             QTextEdit, QLineEdit,
                             QPushButton, QCheckBox,
                             QGridLayout, QVBoxLayout,
                             QHBoxLayout, QMessageBox,
                             QComboBox, QSpinBox, QTextEdit)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt

import numpy as np

from .solution import SolutionGUI
from .seqsolution import SeqSolutionGUI
from .. import logmaker
from .. import converters


class EquilibriumCreationError(Exception):
    pass


class SolverGUI(QWidget):
    def __init__(self, eqsys, type_eq, parent=None):
        super().__init__(parent)
        self.parent_ = parent #TODO: There must be some PyQt actual solution
        self.eqsys = eqsys
        self.type_eq = type_eq
        self.solid_phases = None
        self.initializeUI()
    
    def initializeUI(self):
        if not self.has_parent:
            self.setGeometry(100, 100, 300, 300)
            self.setWindowTitle("PyEquion GUI Solver")
        self.setupWidgets()
        self.show()
        
    def setupWidgets(self):
        nonbox_layout = QHBoxLayout()
        settings_layout = QVBoxLayout()
        components_layout = QVBoxLayout()
        
        gen_settings_label = QLabel("Settings")
        gen_settings_label.setAlignment(Qt.AlignCenter)
        activity_hbox = QHBoxLayout()
        activity_label = QLabel("Activity model")
        self.activity_cbox = QComboBox()
        self.activity_cbox.addItems(["IDEAL", "DEBYE", "EXTENDED_DEBYE", "PITZER"])
        self.activity_cbox.setCurrentIndex(3)
        self.activity_cbox.currentIndexChanged.connect(self.recalculate_activity_function)
        self.water_activity_checkbox = QCheckBox("Water activity")
        self.water_activity_checkbox.setChecked(False)
        self.water_activity_checkbox.stateChanged.connect(self.recalculate_activity_function)
        activity_hbox.addWidget(activity_label)
        activity_hbox.addWidget(self.activity_cbox)
        activity_hbox.addWidget(self.water_activity_checkbox)
        
        temp_hbox = QHBoxLayout()
        temp_label = QLabel("Temperature")
        self.temp_value_ledit = QLineEdit()
        self.temp_value_ledit.setText("298.15")
        self.temp_unit_cbox = QComboBox()
        self.temp_unit_cbox.addItems(["K", "ºC", "ºF"])
        self.temp_unit_cbox.setCurrentIndex(0)
        temp_hbox.addWidget(temp_label)
        temp_hbox.addWidget(self.temp_value_ledit)
        temp_hbox.addWidget(self.temp_unit_cbox)

        pressure_hbox = QHBoxLayout()
        pressure_label = QLabel("Pressure")
        self.pressure_value_ledit = QLineEdit()
        self.pressure_value_ledit.setText("1")
        self.pressure_unit_cbox = QComboBox()
        self.pressure_unit_cbox.addItems(["atm", "bar", "Pa"])
        self.pressure_unit_cbox.setCurrentIndex(0)
        pressure_hbox.addWidget(pressure_label)
        pressure_hbox.addWidget(self.pressure_value_ledit)
        pressure_hbox.addWidget(self.pressure_unit_cbox)
        
        type_equilibrium_marker = "aqueous only" if (self.type_eq == "aqueous") else "phases precipitation"
        type_equilibrium_string = "Equilibrium type: {0}".format(type_equilibrium_marker)
        type_equilibrium_label = QLabel(type_equilibrium_string)
        #type_equilibrium_label = QLabel()
        settings_layout.addLayout(temp_hbox)
        settings_layout.addLayout(pressure_hbox)
        settings_layout.addLayout(activity_hbox)
        settings_layout.addWidget(type_equilibrium_label)
        settings_layout.addStretch()
        settings_layout.setContentsMargins(5, 5, 25, 5)

        self.comps_pairs = []
        if self.type_eq == "aqueous":
            comps_labels = self.eqsys.solute_elements + self.eqsys.solutes
        elif self.type_eq == "phase":
            comps_labels = self.eqsys.solute_elements
        comps_vbox = QVBoxLayout()
        comps_title = QLabel("Components values")
        comps_title.setAlignment(Qt.AlignCenter)
        comps_vbox.addWidget(comps_title)
        for i in range(self.eqsys.nsolelements):
            hbox_layout = QHBoxLayout()
            comps_cbox = QComboBox()
            comps_cbox.addItems(comps_labels)
            comps_cbox.setCurrentIndex(i)
            comps_line_edit = QLineEdit()
            comps_line_edit.setFixedWidth(50)
            comps_unit_cbox = QComboBox()
            comps_unit_cbox.addItems(["Molal"])
            comps_unit_cbox.addItems(["mg/L"])
            comps_unit_cbox.addItems(["mM"])
            comps_unit_cbox.addItems(["pp"])
            comps_log_checkbox = QCheckBox("log")
            comps_log_checkbox.setChecked(False)
            act_checkbox = QCheckBox("act")
            act_checkbox.setChecked(False)
            hbox_layout.addWidget(comps_cbox)
            hbox_layout.addWidget(comps_line_edit)
            hbox_layout.addWidget(comps_unit_cbox)
            hbox_layout.addWidget(comps_log_checkbox)
            hbox_layout.addWidget(act_checkbox)
            comps_vbox.addLayout(hbox_layout)
            self.comps_pairs.append((comps_cbox, comps_line_edit,
                                     comps_unit_cbox, comps_log_checkbox, act_checkbox))
        
        closing_equation_hbox = QHBoxLayout()
        closing_equation_label = QLabel("Closing condition")
        self.closing_equation_cbox = QComboBox()
        self.closing_equation_cbox.addItems(["Electroneutrality"]) #TODO: Add others
        closing_equation_hbox.addWidget(closing_equation_label)
        closing_equation_hbox.addWidget(self.closing_equation_cbox)

        self.solid_phases_button = QPushButton("Set solid phases")
        self.solid_phases_button.clicked.connect(self.set_solid_phases)
        
        sequencer_hbox = QHBoxLayout()
        #self.sequencer_checkbox = QCheckBox("Calculate sequence")
        #self.sequencer_checkbox.setLayoutDirection(Qt.RightToLeft)
        sequencer_number_points_label = QLabel("Number of points")
        self.sequencer_number_points_spinbox = QSpinBox()
        self.sequencer_number_points_spinbox.setMinimum(2)
        self.sequencer_number_points_spinbox.setValue(11)
        #sequencer_hbox.addWidget(self.sequencer_checkbox)
        sequencer_hbox.addWidget(sequencer_number_points_label)
        sequencer_hbox.addWidget(self.sequencer_number_points_spinbox)        
        
        self.calculate_button = QPushButton("Calculate equilibrium")
        self.calculate_button.clicked.connect(self.calculate_equilibrium)
        components_layout.addLayout(comps_vbox)
        components_layout.addLayout(closing_equation_hbox)
        if self.type_eq == "phase":
            components_layout.addWidget(self.solid_phases_button)
        components_layout.addLayout(sequencer_hbox)
        components_layout.addStretch()
        components_layout.setContentsMargins(25, 5, 5, 5)

        nonbox_layout.addLayout(settings_layout)        
        nonbox_layout.addLayout(components_layout)
        nonbox_layout.setContentsMargins(10, 10, 10, 10)
        
        main_layout = QVBoxLayout()
        main_layout.addLayout(nonbox_layout)
        main_layout.addWidget(self.calculate_button)
        
        self.setLayout(main_layout)
        
    def recalculate_activity_function(self):
        activity_function = self.activity_cbox.currentText()
        water_activity = self.water_activity_checkbox.isChecked()
        self.eqsys.set_activity_functions(activity_function, water_activity)
        
    def calculate_equilibrium(self):
        #is_sequence = self.sequencer_checkbox.isChecked()
        #npoints = self.sequencer_number_points_spinbox.value() if is_sequence else None
        try:
            temperature = self.get_temperature()
            pressure = self.get_pressure()
            molal_balance, activity_balance, \
            molal_balance_log, activity_balance_log = self.get_balances(temperature, pressure)
            closing_equation, closing_equation_value = self.get_closing_conditions()
            pairs = self.get_pairs(molal_balance, activity_balance, molal_balance_log, activity_balance_log,
                                   temperature, pressure, closing_equation, closing_equation_value)
            is_sequence = len(pairs) > 0
            npoints = self.sequencer_number_points_spinbox.value() if is_sequence else None
        except EquilibriumCreationError:
            return
        solver_log = logmaker.make_solver_log(
                        molal_balance, activity_balance,
                        molal_balance_log, activity_balance_log,
                        temperature, pressure,
                        closing_equation, closing_equation_value,
                        npoints=npoints)
        try:
            if not is_sequence:
                if self.type_eq == "aqueous":
                    solution, stats = self.eqsys.solve_equilibrium_mixed_balance(
                                            temperature,
                                            molal_balance,
                                            activity_balance,
                                            molal_balance_log,
                                            activity_balance_log,
                                            closing_equation,
                                            closing_equation_value,
                                            pressure)
                elif self.type_eq == "phase":
                    solution, stats = self.eqsys.solve_equilibrium_elements_balance_phases(
                                            temperature, molal_balance,
                                            solid_phases=self.solid_phases,
                                            PATM=pressure)
            else:
                if self.type_eq == "aqueous":
                    solution, stats = self.eqsys.solve_equilibrium_mixed_balance_sequential(
                                              temperature,
                                              molal_balance,
                                              activity_balance,
                                              molal_balance_log,
                                              activity_balance_log,
                                              closing_equation,
                                              closing_equation_value,
                                              pressure,
                                              npoints=npoints)
                elif self.type_eq == "phase":
                    solution, stats = self.eqsys.solve_equilibrium_elements_balance_phases_sequential(
                                              temperature, molal_balance,
                                              PATM=pressure,
                                              solid_phases=self.solid_phases,
                                              npoints=npoints)

        except: #Generic something happened
            QMessageBox.critical(self, 
                                 "Could not complete calculation",
                                 "Could not complete calculation. "\
                                 "Try to either change the activity model "\
                                 "or the solver settings.",
                                 QMessageBox.Close,
                                 QMessageBox.Close)
            return
        res_num = np.max(np.abs(np.array(stats['res']))) #Valid for both cases
        QMessageBox.information(self,
                                "Calculation successful",
                                "Calculation successful. "\
                                "The residual was {:.3e}".format(res_num),
                                QMessageBox.Ok,
                                QMessageBox.Ok)
        if not is_sequence:
            solution_gui = SolutionGUI(solution, solver_log, self.type_eq, self.parent_)
        else:
            solution_gui = SeqSolutionGUI(solution, solver_log, self.type_eq, pairs, self.parent_)
        self.create_new_gui(solution_gui)

    def set_solid_phases(self):
        self.solid_gui = SolidPhasesGUI(self)
        #self.solid_gui.show()
    
    #FIXME: in get_balances, get_temperature and get_pressure there is severe boilerplating
    def get_balances(self, temperature, pressure):
        molal_balance = dict()
        activity_balance = dict()
        molal_balance_log = dict()
        activity_balance_log = dict()
        return_list = [molal_balance,
                        activity_balance,
                        molal_balance_log,
                        activity_balance_log]
        for comp_cbox, line_edit, unit_cbox, log_checkbox, act_checkbox in \
            self.comps_pairs:
            is_log = log_checkbox.isChecked()
            is_activity = act_checkbox.isChecked()
            comp = comp_cbox.currentText()
            conv_factor = self.convert_comp(comp, unit_cbox.currentText(), temperature)
            try:
                text = line_edit.text()
                text = ''.join(text.split()) #HACK
                if "," in text:
                    val = tuple(map(lambda x : conv_factor*float(x),
                                    text.split(",")))
                else:
                    val = conv_factor*float(text)
            except ValueError:
                self.show_creation_error("Some component was set to non-numerical value "\
                                         "or is not positive")
            if is_activity:
                if comp in self.eqsys.solute_elements:
                    self.show_creation_error("Element balance can't be in activity")
            if not is_log:
                if (logmaker.is_number(val) and val <= 0) or \
                    (logmaker.is_sequence(val) and (val[0]*val[1] <= 0)):
                    self.show_creation_error("Only balance in logs can be set to negative")
            if is_log and self.type_eq == "phase":
                self.show_creation_error("In phases mode, elements can't be set in log")
            if comp in itertools.chain(*return_list):
                self.show_creation_error("Repeated components")
            if not is_activity and not is_log:
                molal_balance[comp] = val
            elif is_activity and not is_log:
                activity_balance[comp] = val
            elif not is_activity and is_log:
                molal_balance_log[comp] = val
            elif is_activity and is_log:
                activity_balance_log[comp] = val
        return return_list
    
    def get_temperature(self):
        TK = None
        try:
            text = self.temp_value_ledit.text()
            text = ''.join(text.split()) #HACK
            if "," in text:
                T = tuple(map(float, text.split(",")))
            else:
                T = float(text)
            T = np.array(T)
        except ValueError:
            self.show_creation_error("Temperature is non-numerical")
        temp_unit = self.temp_unit_cbox.currentText()
        if temp_unit == "K":
            TK = T
        elif temp_unit == "ºC":
            TK = T + 273.15
        elif temp_unit == "ºF": 
            TK = (32.*T - 32.)*5/9 + 273.15
        # if TK < 0:
        #     self.show_creation_error("Temperature below 0K")
        try:
            TK = float(TK)
        except:
            TK = tuple(TK)
        return TK
        
    def get_pressure(self):
        pressure = None
        try:
            text = self.pressure_value_ledit.text()
            text = ''.join(text.split()) #HACK
            if "," in text:
                pressure = tuple(map(float, text.split(",")))
            else:
                pressure = float(text)
            pressure = np.array(pressure)
        except ValueError:
            self.show_creation_error("Pressure is non-numerical")
        # if pressure <= 0:
        #     self.show_creation_error("Pressure is non-positive")
        pressure_unit = self.pressure_unit_cbox.currentText()
        if pressure_unit == "atm":
            pressure = pressure #No change
        elif pressure_unit == "bar":
            pressure = pressure*0.98692326671
        elif pressure_unit == "Pa":
            pressure = pressure*0.98692326671*1e-5
        try:
            pressure = float(pressure)
        except:
            pressure = tuple(pressure)
        return pressure
    
    def get_pairs(self, molal_balance, activity_balance, molal_balance_log, activity_balance_log,
                  temperature, pressure, closing_equation, closing_equation_value):
        pairs = []
        pair_tuple = collections.namedtuple("Pair", "name bounds unit")
        if isinstance(temperature, tuple):
            pairs.append(pair_tuple("T", temperature, "K"))
        if isinstance(pressure, tuple):
            pairs.append(pair_tuple("P", pressure, "atm"))
        if isinstance(closing_equation_value, tuple):
            pairs.append(pair_tuple(closing_equation, closing_equation_value, " "))
        for key, value in molal_balance.items():
            if isinstance(value, tuple):
                pairs.append(pair_tuple("[{0}]".format(key), value, "mol/kg H2O"))
        for key, value in activity_balance.items():
            if isinstance(value, tuple):
                pairs.append(pair_tuple("{{{0}}}".format(key), value, "mol/kg H2O"))
        for key, value in molal_balance_log.items():
            if isinstance(value, tuple):
                pairs.append(pair_tuple("log[{0}]".format(key), value, "mol/kg H2O"))
        for key, value in activity_balance_log.items():
            if isinstance(value, tuple):
                pairs.append(pair_tuple("log{{{0}}}".format(key), value, "mol/kg H2O"))
        return pairs
    
    def convert_comp(self, comp_val, comp_unit, temperature):
        if comp_unit == "Molal":
            return 1.0
        elif comp_unit == "mM":
            return converters.mmolar_to_molal(1.0, temperature) #FIXME
        elif comp_unit == "mg/L":
            return converters.mgl_to_molal(1.0, comp_val, temperature)
        elif comp_unit == "PP":
            raise converters.get_activity_from_partial_pressure(1.0, comp_val, temperature)
    def get_closing_conditions(self):
        return "electroneutrality", 0.0 #TODO: put more
    
    def show_creation_error(self, text):
        QMessageBox.critical(self, 
                             "Could not initiate calculation",
                             text,
                             QMessageBox.Close,
                             QMessageBox.Close)
        raise EquilibriumCreationError

    def create_new_gui(self, new_gui):
        if not self.has_parent:
            self.new_gui = new_gui
            self.new_gui.show()
        else:
            self.parent_.display_and_connect(self, new_gui, "Solution")

    @property
    def has_parent(self):
        return self.parent_ is not None


class SolidPhasesGUI(QWidget):
    def __init__(self, parent=None):
        super().__init__()
        self.parent_ = parent #TODO: There must be some PyQt actual solution
        self.initializeUI()
    
    def initializeUI(self):
        self.setGeometry(100, 100, 300, 300)
        self.setWindowTitle('Solid phases')
        self.setupWidgets()
        self.show()

    def setupWidgets(self):
        main_layout = QVBoxLayout()
        elements_label = QLabel("Solid phases")
        self.inserter = QTextEdit()
        placeholder_text = "Set every solid phase in a row. Possible solids : \n"
        placeholder_text += "\n".join(self.parent_.eqsys.solid_phase_names)
        self.inserter.setPlaceholderText(placeholder_text)
        self.ok_button = QPushButton("Ok")
        self.ok_button.clicked.connect(self.accept)
        main_layout.addWidget(elements_label)
        main_layout.addWidget(self.inserter)
        main_layout.addWidget(self.ok_button)
        self.setLayout(main_layout)
        
    def accept(self):
        solid_phases = [s.strip() for s in self.inserter.toPlainText().strip('\n').split('\n')]
        for s in solid_phases:
            if s not in self.parent_.eqsys.solid_phase_names:
                QMessageBox.critical(self, 
                                     "Non valid solid",
                                     "%s is not a possible solid. "%s + \
                                     "Setting solid phases automatically",
                                     QMessageBox.Close,
                                     QMessageBox.Close)
                solid_phases = []
                break
        QMessageBox.information(self,
                                "Solid phases setted.",
                                "Solid phases setted.",
                                QMessageBox.Ok,
                                QMessageBox.Ok)

        if len(solid_phases) == 0:
            self.parent_.solid_phases = None
        else:
            self.parent_.solid_phases = solid_phases
        print(self.parent_.solid_phases)
    
