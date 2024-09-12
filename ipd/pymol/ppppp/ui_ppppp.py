# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'form.ui'
##
## Created by: Qt User Interface Compiler version 6.7.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QPushButton, QSizePolicy, QWidget)

class Ui_PPPPP_GUI(object):
    def setupUi(self, PPPPP_GUI):
        if not PPPPP_GUI.objectName():
            PPPPP_GUI.setObjectName(u"PPPPP_GUI")
        PPPPP_GUI.resize(157, 343)
        self.stier = QPushButton(PPPPP_GUI)
        self.stier.setObjectName(u"stier")
        self.stier.setGeometry(QRect(40, 30, 80, 25))
        self.atier = QPushButton(PPPPP_GUI)
        self.atier.setObjectName(u"atier")
        self.atier.setGeometry(QRect(40, 80, 80, 25))
        self.btier = QPushButton(PPPPP_GUI)
        self.btier.setObjectName(u"btier")
        self.btier.setGeometry(QRect(40, 130, 80, 25))
        self.ctier = QPushButton(PPPPP_GUI)
        self.ctier.setObjectName(u"ctier")
        self.ctier.setGeometry(QRect(40, 180, 80, 25))
        self.dtier = QPushButton(PPPPP_GUI)
        self.dtier.setObjectName(u"dtier")
        self.dtier.setGeometry(QRect(40, 230, 80, 25))
        self.ftier = QPushButton(PPPPP_GUI)
        self.ftier.setObjectName(u"ftier")
        self.ftier.setGeometry(QRect(40, 280, 80, 25))

        self.retranslateUi(PPPPP_GUI)

        QMetaObject.connectSlotsByName(PPPPP_GUI)
    # setupUi

    def retranslateUi(self, PPPPP_GUI):
        PPPPP_GUI.setWindowTitle(QCoreApplication.translate("PPPPP_GUI", u"PPPPP_GUI", None))
#if QT_CONFIG(tooltip)
        self.stier.setToolTip(QCoreApplication.translate("PPPPP_GUI", u"<html><head/><body><p>This is the most beautiful protein you've ever seen.</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.stier.setText(QCoreApplication.translate("PPPPP_GUI", u"S Tier", None))
#if QT_CONFIG(tooltip)
        self.atier.setToolTip(QCoreApplication.translate("PPPPP_GUI", u"<html><head/><body><p>This is a nice design. 9/10 would order again.</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.atier.setText(QCoreApplication.translate("PPPPP_GUI", u"A Tier", None))
#if QT_CONFIG(tooltip)
        self.btier.setToolTip(QCoreApplication.translate("PPPPP_GUI", u"<html><head/><body><p>You might order this protein.</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.btier.setText(QCoreApplication.translate("PPPPP_GUI", u"B Tier", None))
#if QT_CONFIG(tooltip)
        self.ctier.setToolTip(QCoreApplication.translate("PPPPP_GUI", u"<html><head/><body><p>You might order this protein... on a chip.</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.ctier.setText(QCoreApplication.translate("PPPPP_GUI", u"C Tier", None))
#if QT_CONFIG(tooltip)
        self.dtier.setToolTip(QCoreApplication.translate("PPPPP_GUI", u"<html><head/><body><p>This is a bad protein. Even David doesn't like it.</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.dtier.setText(QCoreApplication.translate("PPPPP_GUI", u"D Tier", None))
#if QT_CONFIG(tooltip)
        self.ftier.setToolTip(QCoreApplication.translate("PPPPP_GUI", u"<html><head/><body><p>This protein is butt ugly.</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.ftier.setText(QCoreApplication.translate("PPPPP_GUI", u"F Tier", None))
    # retranslateUi

