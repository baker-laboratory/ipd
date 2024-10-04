from ipd.qt.context_menu import *
from ipd.dev import lazyimport, printed_string

Qt = lazyimport('pymol.Qt')

def widget_gettext(widget):
    if hasattr(widget, 'text'): return widget.text()
    if hasattr(widget, 'toPlainText'): return widget.toPlainText()
    if hasattr(widget, 'currentText'): return widget.currentText()

def notify(message):
    message = printed_string(message)
    print('NOTIFY:', message)
    Qt.QtWidgets.QMessageBox.warning(None, "Warning", message)

def isfalse_notify(ok, message):
    if not ok:
        Qt.QtWidgets.QMessageBox.warning(None, "Warning", message)
        return True
