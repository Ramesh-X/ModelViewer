'''
Created on Jun 18, 2018

@author: rameshpr
'''
from PyQt4 import QtGui
import sys
from ui import MainWindow

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    form = MainWindow()
    form.show()
    app.exec_()