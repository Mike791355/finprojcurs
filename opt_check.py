# Form implementation generated from reading ui file 'Opt_check.ui'
#
# Created by: PyQt6 UI code generator 6.4.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_opt_check(object):
    def setupUi(self, opt_check):
        opt_check.setObjectName("opt_check")
        opt_check.resize(400, 300)
        self.gridLayout = QtWidgets.QGridLayout(opt_check)
        self.gridLayout.setObjectName("gridLayout")
        self.label = QtWidgets.QLabel(parent=opt_check)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 2, 3)
        self.opt_good = QtWidgets.QPushButton(parent=opt_check)
        self.opt_good.setObjectName("pushButton")
        self.gridLayout.addWidget(self.opt_good, 2, 0, 1, 1)
        self.opt_bad = QtWidgets.QPushButton(parent=opt_check)
        self.opt_bad.setObjectName("pushButton_2")
        self.gridLayout.addWidget(self.opt_bad, 2, 2, 1, 1)

        self.retranslateUi(opt_check)
        QtCore.QMetaObject.connectSlotsByName(opt_check)

    def retranslateUi(self, opt_check):
        _translate = QtCore.QCoreApplication.translate
        opt_check.setWindowTitle(_translate("opt_check", "Dialog"))
        self.label.setText(_translate("opt_check", "Как вы оцениваете предложенную оптимизацию ?"))
        self.opt_good.setText(_translate("opt_check", "Хорошо"))
        self.opt_bad.setText(_translate("opt_check", "Плохо"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    opt_check = QtWidgets.QDialog()
    ui = Ui_opt_check()
    ui.setupUi(opt_check)
    opt_check.show()
    sys.exit(app.exec())
