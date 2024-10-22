from ipd.dev import lazyimport, printed_string
from ipd.dev.qt.context_menu import *

Qt = lazyimport('pymol.Qt')

def widget_gettext(widget):
    if hasattr(widget, 'text'): return widget.text()
    if hasattr(widget, 'toPlainText'): return widget.toPlainText()
    if hasattr(widget, 'currentText'): return widget.currentText()
    raise ValueError(f'cant get text from widget {widget}')

def widget_settext(widget, text):
    if hasattr(widget, 'setText'): return widget.setText(text)
    if hasattr(widget, 'setPlainText'): return widget.setPlainText(text)
    raise ValueError(f'cant set text from widget {widget}')

def notify(message):
    message = printed_string(message)
    print('NOTIFY:', message)
    Qt.QtWidgets.QMessageBox.warning(None, "Warning", message)

def isfalse_notify(ok, message):
    if not ok:
        Qt.QtWidgets.QMessageBox.warning(None, "Warning", message)
        return True
