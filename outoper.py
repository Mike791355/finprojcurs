# Form implementation generated from reading ui file 'outoper.ui'
#
# Created by: PyQt6 UI code generator 6.4.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class OutUi_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(480, 300)
        self.gridLayout_2 = QtWidgets.QGridLayout(Dialog)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.frame = QtWidgets.QFrame(parent=Dialog)
        self.frame.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
        self.frame.setObjectName("frame")
        self.gridLayout = QtWidgets.QGridLayout(self.frame)
        self.gridLayout.setObjectName("gridLayout")
        self.outctg = QtWidgets.QComboBox(parent=self.frame)
        self.outctg.setObjectName("outctg")
        self.outctg.addItem("")
        self.outctg.addItem("")
        self.outctg.addItem("")
        self.outctg.addItem("")
        self.outctg.addItem("")
        self.outctg.addItem("")
        self.outctg.addItem("")
        self.outctg.addItem("")
        self.outctg.addItem("")
        self.gridLayout.addWidget(self.outctg, 3, 1, 1, 1)
        self.label_4 = QtWidgets.QLabel(parent=self.frame)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 3, 5, 1, 1)
        self.label_3 = QtWidgets.QLabel(parent=self.frame)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 3, 0, 1, 1)
        self.outtran_sum = QtWidgets.QLineEdit(parent=self.frame)
        self.outtran_sum.setObjectName("outtran_sum")
        self.gridLayout.addWidget(self.outtran_sum, 3, 3, 1, 1)
        self.outtr_confirm = QtWidgets.QPushButton(parent=self.frame)
        self.outtr_confirm.setObjectName("outtr_confirm")
        self.gridLayout.addWidget(self.outtr_confirm, 4, 2, 1, 2)
        self.label = QtWidgets.QLabel(parent=self.frame)
        self.label.setStyleSheet("font: 15pt \"Segoe UI\";\n"
"font: 16pt \"Segoe UI\";")
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 3)
        self.outtr_desc = QtWidgets.QLineEdit(parent=self.frame)
        self.outtr_desc.setObjectName("outtr_desc")
        self.gridLayout.addWidget(self.outtr_desc, 3, 6, 1, 1)
        self.label_2 = QtWidgets.QLabel(parent=self.frame)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 3, 2, 1, 1)
        self.outtr_date = QtWidgets.QDateEdit(parent=self.frame)
        self.outtr_date.setObjectName("outtr_date")
        self.gridLayout.addWidget(self.outtr_date, 0, 5, 1, 2)
        self.gridLayout_2.addWidget(self.frame, 0, 0, 1, 1)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.outctg.setItemText(0, _translate("Dialog", "Ком.платежи и сборы"))
        self.outctg.setItemText(1, _translate("Dialog", "Транспорт"))
        self.outctg.setItemText(2, _translate("Dialog", "Здоровье и фитнес"))
        self.outctg.setItemText(3, _translate("Dialog", "Еда"))
        self.outctg.setItemText(4, _translate("Dialog", "Покупки"))
        self.outctg.setItemText(5, _translate("Dialog", "Подписки"))
        self.outctg.setItemText(6, _translate("Dialog", "Путешествия"))
        self.outctg.setItemText(7, _translate("Dialog", "Развлечения"))
        self.outctg.setItemText(8, _translate("Dialog", "Другое"))
        self.label_4.setText(_translate("Dialog", "Описание"))
        self.label_3.setText(_translate("Dialog", "Категория"))
        self.outtr_confirm.setText(_translate("Dialog", "Подтвердить "))
        self.label.setText(_translate("Dialog", "Транзакция (Расход)"))
        self.label_2.setText(_translate("Dialog", "Сумма"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = OutUi_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec())
