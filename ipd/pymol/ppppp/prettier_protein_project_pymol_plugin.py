import os
import sys

def record_result(ui, grade, *args):
    print('grade')

def run_ppppp_gui(_self=None):
    if _self is None:
        from pymol import cmd as _self

    from pymol.Qt import QtWidgets
    from pymol.Qt.utils import loadUi

    dialog = QtWidgets.QDialog()
    uifile = os.path.join(os.path.dirname(__file__), 'ppppp.ui')
    ui = loadUi(uifile, dialog)
    ui.stier.clicked.connect(lambda *a: record_result(ui, 's', *a))
    ui.atier.clicked.connect(lambda *a: record_result(ui, 'a', *a))
    ui.btier.clicked.connect(lambda *a: record_result(ui, 'b', *a))
    ui.ctier.clicked.connect(lambda *a: record_result(ui, 'c', *a))
    ui.dtier.clicked.connect(lambda *a: record_result(ui, 'd', *a))
    ui.ftier.clicked.connect(lambda *a: record_result(ui, 'f', *a))
    ui.quit.clicked.connect(sys.exit)
    dialog.show()
