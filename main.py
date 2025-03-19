import sys

from PyQt6 import QtWidgets
from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.QtSql import QSqlTableModel

from finaiui import Ui_MainWindow
from bdfinai import bdmain
from inoper import InUi_Dialog
from outoper import OutUi_Dialog

tr = "default_type"

class finlogic(QMainWindow):
    def __init__(self):
        super(finlogic, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.conn = bdmain()
        #self.view_data()
        self.reload_data()

        self.ui.intrans.clicked.connect(self.inoper_w)
        self.ui.outtrans.clicked.connect(self.outoper_w)

    def reload_data(self):
        self.ui.totalin.setText(self.conn.total_income())
        self.ui.totalout.setText(self.conn.total_outcome())
        self.ui.bills_spent.setText(self.conn.total_bills())
        self.ui.transp_spent.setText(self.conn.total_transport())

    #def view_data(self):
    #    self.model = QSqlTableModel(self)
    #    self.model.setTable('expenses')
    #    self.model.select()
    #    self.ui.tableView.setModel(self.model)

    def inoper_w(self):
        self.new_window = QtWidgets.QDialog()
        self.ui_window = InUi_Dialog()
        self.ui_window.setupUi(self.new_window)
        self.new_window.show()
        sender = self.sender()
        self.tr = "Income"
        self.ui_window.intr_confirm.clicked.connect(self.add_new_transaction)

    def outoper_w(self):
        self.new_window = QtWidgets.QDialog()
        self.ui_window = OutUi_Dialog()
        self.ui_window.setupUi(self.new_window)
        self.new_window.show()
        sender = self.sender()
        self.tr = "Outcome"
        self.ui_window.outtr_confirm.clicked.connect(self.add_new_transaction)

    def add_new_transaction(self):
        if self.tr == "Income":
            date = self.ui_window.intran_date.text()
            category = self.ui_window.intran_ctg.currentText()
            description = self.ui_window.intran_desc.text()
            balance = self.ui_window.intran_sum.text()
            tr_type = self.tr

            self.conn.add_new_transaction_query(
                tr_type, category, balance, description, date
            )
            self.reload_data()
            self.new_window.close()
        elif self.tr == "Outcome":
            date = self.ui_window.outtr_date.text()
            category = self.ui_window.outctg.currentText()
            description = self.ui_window.outtr_desc.text()
            balance = self.ui_window.outtran_sum.text()
            tr_type = self.tr

            self.conn.add_new_transaction_query(
                tr_type, category, balance, description, date
            )
            self.reload_data()
            self.new_window.close()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = finlogic()
    window.show()

    sys.exit(app.exec())