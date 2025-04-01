from PyQt6 import QtCore, QtGui, QtWidgets


class InUi_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(532, 300)
        self.gridLayout_2 = QtWidgets.QGridLayout(Dialog)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.frame = QtWidgets.QFrame(parent=Dialog)
        self.frame.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
        self.frame.setObjectName("frame")
        self.gridLayout = QtWidgets.QGridLayout(self.frame)
        self.gridLayout.setObjectName("gridLayout")
        self.intran_ctg = QtWidgets.QComboBox(parent=self.frame)
        self.intran_ctg.setObjectName("intran_ctg")
        self.intran_ctg.addItem("")
        self.intran_ctg.addItem("")
        self.intran_ctg.addItem("")
        self.gridLayout.addWidget(self.intran_ctg, 1, 3, 1, 1)
        self.intr_confirm = QtWidgets.QPushButton(parent=self.frame)
        self.intr_confirm.setObjectName("intr_confirm")
        self.gridLayout.addWidget(self.intr_confirm, 5, 2, 1, 2)
        self.label = QtWidgets.QLabel(parent=self.frame)
        self.label.setStyleSheet("font: 15pt \"Segoe UI\";\n"
"font: 16pt \"Segoe UI\";")
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 3)
        self.label_2 = QtWidgets.QLabel(parent=self.frame)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 3, 2, 1, 1)
        self.intran_sum = QtWidgets.QLineEdit(parent=self.frame)
        self.intran_sum.setObjectName("intran_sum")
        self.gridLayout.addWidget(self.intran_sum, 3, 3, 1, 1)
        self.intran_date = QtWidgets.QDateEdit(parent=self.frame)
        self.intran_date.setObjectName("intran_date")
        self.gridLayout.addWidget(self.intran_date, 0, 5, 1, 2)
        self.intran_desc = QtWidgets.QLineEdit(parent=self.frame)
        self.intran_desc.setObjectName("intran_desc")
        self.gridLayout.addWidget(self.intran_desc, 4, 3, 1, 1)
        self.label_4 = QtWidgets.QLabel(parent=self.frame)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 4, 2, 1, 1)
        self.label_3 = QtWidgets.QLabel(parent=self.frame)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 1, 2, 1, 1)
        self.gridLayout_2.addWidget(self.frame, 0, 0, 1, 1)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.intran_ctg.setItemText(0, _translate("Dialog", "Основной доход"))
        self.intran_ctg.setItemText(1, _translate("Dialog", "Пассивный доход/соц.выплаты"))
        self.intran_ctg.setItemText(2, _translate("Dialog", "Другое"))
        self.intr_confirm.setText(_translate("Dialog", "Подтвердить "))
        self.label.setText(_translate("Dialog", "Транзакция (Доход)"))
        self.label_2.setText(_translate("Dialog", "Сумма"))
        self.label_4.setText(_translate("Dialog", "Описание"))
        self.label_3.setText(_translate("Dialog", "Категория"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = InUi_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec())
