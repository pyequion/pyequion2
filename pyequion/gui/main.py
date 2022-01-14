# -*- coding: utf-8 -*-
import sys
import collections

from PyQt5.QtWidgets import (QApplication, QWidget, QLabel,
                             QTextEdit, QLineEdit,
                             QPushButton, QCheckBox,
                             QGridLayout, QVBoxLayout,
                             QHBoxLayout, QMessageBox,
                             QComboBox, QMainWindow,
                             QTabWidget, QTabBar,
                             QScrollArea, QAction,
                             QFileDialog)
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtCore import Qt

import cloudpickle

from .. import EquilibriumSystem
from .solver import SolverGUI
from .initializer import InitializerGUI
from .solution import SolutionGUI


class PyEquionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initializeUI()
    
    def initializeUI(self):
        self.setGeometry(100, 100, 900, 600)
        self.setWindowTitle('PyEquion GUI')
        self.setCentralWidget(WelcomeWidget(self))
        self.create_menu()
        self.tabWidget = None
        self.show()
        
    def create_menu(self):
        exit_act = QAction('Exit', self)
        exit_act.setShortcut('Ctrl+Q')
        exit_act.triggered.connect(self.close)
        
        open_act = QAction('Open', self)
        open_act.setShortcut('Ctrl+O')
        open_act.triggered.connect(self.close)

        save_act = QAction('Save', self)
        save_act.setShortcut('Ctrl+S')
        save_act.triggered.connect(self.saveToFile)

        self.menu_bar = self.menuBar()
        self.menu_bar.setNativeMenuBar(False)
        
        file_menu = self.menu_bar.addMenu('File')
        file_menu.addAction(open_act)
        file_menu.addAction(save_act)
        file_menu.addSeparator()
        file_menu.addAction(exit_act)

        
    def initialize(self):
        self.tabWidget = TabController(self)
        self.setCentralWidget(self.tabWidget)
    
    def load(self, filename):
        with open(filename, "rb") as f:
            loaded_widget = cloudpickle.load(f)
        self.tabWidget = loaded_widget
        self.setCentralWidget(self.tabWidget)

    def closeEvent(self, event):
        """
        Display a QMessageBox when asking the user if they want to 
        quit the program. 
        """
        answer = QMessageBox.question(self, "Quit PyEquion?",
            "Are you sure you want to Quit?", QMessageBox.No | QMessageBox.Yes, 
            QMessageBox.Yes)
        if answer == QMessageBox.Yes:
            event.accept() # accept the event and close the application
        else:
            event.ignore() # ignore the close event
        
    def saveToFile(self):
        if self.tabWidget is None:
            QMessageBox.information(self, "Error", 
                "Nothing to be saved.", QMessageBox.Ok)
            return

        #else
        file_name, _ = QFileDialog.getSaveFileName(self, 'Save File',
            "","Picke File (*.pkl)")
        if file_name.endswith('.pkl'):
            self.tabWidget.save(file_name)
        else:
            QMessageBox.information(self, "Error", 
                "Unable to save file.", QMessageBox.Ok)

    def setupWidgets(self):
        pass


class TabController(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.nchildren = collections.defaultdict(lambda : 0)
        self.tags = collections.defaultdict(lambda : "1")
        
        self.layout = QVBoxLayout(self)
        # Initialize tab screen
        self.tabs = QTabWidget()
        self.tabs.setTabsClosable(True)
        self.tabs.tabCloseRequested.connect(self.close_tab)
        
        self.initializer_widget = InitializerGUI(self)
        self.tabs.addTab(self.initializer_widget, "Initializer 1")
        self.tabs.tabBar().setTabButton(0, QTabBar.RightSide, None)
        
        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)
        
    def close_tab(self, index):
        widget = self.tabs.widget(index)
        widget.close()
        self.tabs.removeTab(index)
        
    def display_and_connect(self, creator_widget,
                            created_widget,
                            name,
                            create_scroll=True):
        #TODO: Put the connect part
        self.nchildren[creator_widget] += 1
        tag = self.tags[creator_widget] + \
              ".{0}".format(self.nchildren[creator_widget])
        self.tags[created_widget] = tag
        name_ = name + " " + tag #TODO: Put tag in names
        if create_scroll:
            scroll = QScrollArea()
            scroll.setWidgetResizable(True) # CRITICAL
            scroll.setWidget(created_widget) # CRITICAL
            index = self.tabs.addTab(scroll, name_)
        else:
            index = self.tabs.addTab(created_widget, name_)
        self.tabs.setCurrentIndex(index)

    def save(self, filename):
        #Saving and loading are done through TabController
        with open(filename, 'wb') as f:
            cloudpickle.dump(self, f)


class WelcomeWidget(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.initializeUI()
    
    def initializeUI(self):
        self.main_layout = QVBoxLayout(self)
        welcome_label = QLabel("Welcome to PyEquion GUI.")
        welcome_label.setAlignment(Qt.AlignCenter)
        logo_label = QLabel()
        logo_label.setPixmap(QPixmap("./images/pyequion_logo.png"))
        #logo_label.setAlignment(Qt.AlignCenter)
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.initialize)
        self.load_button = QPushButton("Load")
        
        self.main_layout.addWidget(welcome_label)
        self.main_layout.addWidget(logo_label)
        self.main_layout.addWidget(self.start_button)
        self.main_layout.addWidget(self.load_button)
        
    def initialize(self):
        self.parent().initialize()
        self.close()
                
        
# class TabTree(object):
#     def __init__(self):
#         self.tree = dict()
    