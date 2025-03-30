import sys
import datetime
import tensorflow as tf
import numpy as np
import pickle
from PyQt6 import QtWidgets, QtSql, QtCore
from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.QtSql import QSqlTableModel

from finaifull import Ui_MainWindow
from bdfinai import bdmain
from inoper import InUi_Dialog
from outoper import OutUi_Dialog
from in_oper_del import In_Ui_Del_Dialog
from out_oper_del import Out_Ui_Del_Dialog
from goal_fix import goalfixui_dialog
from Opt_check import Ui_opt_check 


with open(r'-', 'rb') as f:  #в кавычках путь к scaler
    scaler = pickle.load(f)
model = tf.keras.models.load_model(r'-', compile=False) # в кавычкх должен быть путь к model.h5 

def get_previous_month(month_str):
    dt = datetime.datetime.strptime(month_str, "%m.%Y")
    year = dt.year
    month = dt.month
    if month == 1:
        prev_month = 12
        prev_year = year - 1
    else:
        prev_month = month - 1
        prev_year = year
    return f"{prev_month:02d}.{prev_year}"

def getlmt(db):
    query = """
        SELECT SUBSTR(Date, 4, 7) as MonthYear 
        FROM transactions 
        ORDER BY SUBSTR(Date, 7, 4) || SUBSTR(Date, 4, 2) DESC 
        LIMIT 1
    """
    q = db.execute_qr(query, [])
    if q and q.next():
        return q.value(0)
    else:
        return None

def getinoutm(db, month_str):
    income_sql = """
        SELECT SUM(Amnt)
        FROM transactions
        WHERE SUBSTR(Date, 4, 7) = ? AND TrType = 'Income'
    """
    income = 0.0
    inq = db.execute_qr(income_sql, [month_str])
    if inq and inq.next():
        inc_val = inq.value(0)
        income = 0.0 if (inc_val is None or inc_val == '') else float(inc_val)
    
    exp_ctg = [
        "Еда",
        "Здоровье и фитнес",
        "Ком.платежи и сборы",
        "Подписки",
        "Покупки",
        "Транспорт",
        "Путешествия",
        "Развлечения"
    ]
    outcomes = []
    outcome_sql = """
        SELECT SUM(Amnt)
        FROM transactions
        WHERE SUBSTR(Date, 4, 7) = ? AND TrType = 'Outcome' AND Category = ?
    """
    for category in exp_ctg:
        qout = db.execute_qr(outcome_sql, [month_str, category])
        outcome_val = 0.0
        if qout and qout.next():
            out_val = qout.value(0)
            outcome_val = 0.0 if (out_val is None or out_val == '') else float(out_val)
        outcomes.append(outcome_val)
    
    return income, outcomes, exp_ctg

ltm = None

class finlogic(QMainWindow):
    def __init__(self):
        super(finlogic, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.conn = bdmain()
        global ltm
        ltm = getlmt(self.conn)
        self.reload_data()

        self.ui.intrans.clicked.connect(self.inoper_w)
        self.ui.outtrans.clicked.connect(self.outoper_w)
        self.ui.pushButton_4.clicked.connect(self.inoperdel_w)
        self.ui.pushButton.clicked.connect(self.outoperdel_w)
        self.ui.set_goal.clicked.connect(self.set_goal_w)
        self.ui.opt_btn.clicked.connect(self.optimize)
        self.ui.opt_btn.clicked.connect(self.opt_check_w)
        self.ui.saveinch.clicked.connect(self.update_incomes_page)
        self.ui.saveoutch.clicked.connect(self.update_expenses_page)

    def reload_data(self):
        current_month = getlmt(self.conn)
        if current_month is None:
            current_month = "01.1970"
        income, outcomes, expense_categories = getinoutm(self.conn, current_month)
        self.ui.totalin.setText(f"Суммарный Доход: {income} ₽")
        total_outcome = sum(outcomes)
        self.ui.totalout.setText(f"Суммарный Расход: {total_outcome} ₽")
        try:
            bills_index = expense_categories.index("Ком.платежи и сборы")
            self.ui.bills_spent.setText(f"Потрачено: {outcomes[bills_index]} ₽")
        except ValueError:
            self.ui.bills_spent.setText("Потрачено: 0 ₽")
        try:
            travs_index = expense_categories.index("Путешествия")
            self.ui.travs_spent.setText(f"Потрачено: {outcomes[travs_index]} ₽")
        except ValueError:
            self.ui.travs_spent.setText("Потрачено: 0 ₽")
        try:
            health_index = expense_categories.index("Здоровье и фитнес")
            self.ui.health_spent.setText(f"Потрачено: {outcomes[health_index]} ₽")
        except ValueError:
            self.ui.health_spent.setText("Потрачено: 0 ₽")
        try:
            transp_index = expense_categories.index("Транспорт")
            self.ui.transp_spent.setText(f"Потрачено: {outcomes[transp_index]} ₽")
        except ValueError:
            self.ui.transp_spent.setText("Потрачено: 0 ₽")
        try:
            shop_index = expense_categories.index("Покупки")
            self.ui.shop_spent.setText(f"Потрачено: {outcomes[shop_index]} ₽")
        except ValueError:
            self.ui.shop_spent.setText("Потрачено: 0 ₽")
        try:
            subs_index = expense_categories.index("Подписки")
            self.ui.subs_spent.setText(f"Потрачено: {outcomes[subs_index]} ₽")
        except ValueError:
            self.ui.subs_spent.setText("Потрачено: 0 ₽")
        try:
            food_index = expense_categories.index("Еда")
            self.ui.health_spent_2.setText(f"Потрачено: {outcomes[food_index]} ₽")
        except ValueError:
            self.ui.health_spent_2.setText("Потрачено: 0 ₽")
        try:
            enj_index = expense_categories.index("Развлечения")
            self.ui.enj_spent.setText(f"Потрачено: {outcomes[enj_index]} ₽")
        except ValueError:
            self.ui.enj_spent.setText("Потрачено: 0 ₽")
        self.ui.nameoutput.setText(self.conn.bdgoal_name())
        self.ui.dateoutput.setText(self.conn.bdgoal_date())
        self.ui.sumoutput.setText(self.conn.bdgoal_amount())
        self.check_and_finetune()
        self.load_transactions()

    def load_transactions(self):
        model = QSqlTableModel(self)
        model.setTable("transactions")
        model.setEditStrategy(QSqlTableModel.EditStrategy.OnFieldChange)
        model.select()
        self.ui.trans_data.setModel(model)

    def update_incomes_page(self):
        
        try:
            main_income = float(self.ui.mainincomeline.text())
        except:
            main_income = 0.0
        try:
            sec_income = float(self.ui.secincomeline.text())
        except:
            sec_income = 0.0
        try:
            oth_income = float(self.ui.othincomeline.text())
        except:
            oth_income = 0.0

        self.ui.label_21.setText(f"Основной Доход; план: {main_income:.2f} руб.")
        self.ui.label_22.setText(f"Пассивный Доход и Соц.выплаты; план: {sec_income:.2f} руб.")
        self.ui.label_26.setText(f"Другое; план: {oth_income:.2f} руб.")
        QtWidgets.QMessageBox.information(self, "Доход", "Данные доходов обновлены в заголовках.")

    def update_expenses_page(self):

        current_month = getlmt(self.conn)
        if current_month is None:
            current_month = "01.2000"
        _, outcomes, expense_categories = getinoutm(self.conn, current_month)
        mapping = {
            "Ком.платежи и сборы": (self.ui.bills_spent, self.ui.bills_line),
            "Путешествия": (self.ui.travs_spent, self.ui.travs_line),
            "Здоровье и фитнес": (self.ui.health_spent, self.ui.health_line),
            "Транспорт": (self.ui.transp_spent, self.ui.transp_line),
            "Покупки": (self.ui.shop_spent, self.ui.shop_line),
            "Подписки": (self.ui.subs_spent, self.ui.subs_line),
            "Еда": (self.ui.health_spent_2, self.ui.food_line),
            "Развлечения": (self.ui.enj_spent, self.ui.enj_line)
        }
        for category, (label_widget, budget_widget) in mapping.items():
            try:
                idx = expense_categories.index(category)
                spent = outcomes[idx]
            except ValueError:
                spent = 0.0
            budget_text = budget_widget.text().strip()
            try:
                budget = float(budget_text)
            except:
                budget = 0.0
            label_widget.setText(f"Потрачено: {spent:.2f} ₽  Бюджет: {budget:.2f} ₽")
        QtWidgets.QMessageBox.information(self, "Бюджет", "Данные бюджета обновлены на странице расходов.")

    def inoper_w(self):
        self.new_window = QtWidgets.QDialog()
        self.ui_window = InUi_Dialog()
        self.ui_window.setupUi(self.new_window)
        self.new_window.show()
        self.tr = "Income"
        self.ui_window.intr_confirm.clicked.connect(self.add_new_transaction)

    def outoper_w(self):
        self.new_window = QtWidgets.QDialog()
        self.ui_window = OutUi_Dialog()
        self.ui_window.setupUi(self.new_window)
        self.new_window.show()
        self.tr = "Outcome"
        self.ui_window.outtr_confirm.clicked.connect(self.add_new_transaction)
    
    def inoperdel_w(self):
        self.new_window = QtWidgets.QDialog()
        self.ui_window = In_Ui_Del_Dialog()
        self.ui_window.setupUi(self.new_window)
        self.new_window.show()
        self.ui_window.indel_confirm.clicked.connect(
            lambda: (
                self.conn.delete_transaction(self.ui_window.indelsum.text()),
                self.reload_data(),
                self.new_window.close()
            )
        )
    
    def outoperdel_w(self):
        self.new_window = QtWidgets.QDialog()
        self.ui_window = Out_Ui_Del_Dialog()
        self.ui_window.setupUi(self.new_window)
        self.new_window.show()
        self.ui_window.outdelconfirm.clicked.connect(
            lambda: (
                self.conn.delete_transaction(self.ui_window.outdelsum.text()),
                self.reload_data(),
                self.new_window.close()
            )
        )

    def set_goal_w(self):
        self.new_window = QtWidgets.QDialog()
        self.ui_window = goalfixui_dialog()
        self.ui_window.setupUi(self.new_window)
        self.new_window.show()
        self.ui_window.goalsetbtn.clicked.connect(
            lambda: (
                self.conn.set_goal(
                    self.ui_window.goalname.text(),
                    self.ui_window.goaldate.text(),
                    float(self.ui_window.goalsum.text())
                ),
                self.reload_data(),
                self.new_window.close()
            )
        )

    def add_new_transaction(self):
        if self.tr == "Income":
            date = self.ui_window.intran_date.text()
            category = self.ui_window.intran_ctg.currentText()
            description = self.ui_window.intran_desc.text()
            balance = self.ui_window.intran_sum.text()
            tr_type = self.tr
            new_id = self.conn.add_new_transaction_query(tr_type, category, balance, description, date)
            self.reload_data()
            self.new_window.close()
        elif self.tr == "Outcome":
            date = self.ui_window.outtr_date.text()
            category = self.ui_window.outctg.currentText()
            description = self.ui_window.outtr_desc.text()
            balance = self.ui_window.outtran_sum.text()
            tr_type = self.tr
            new_id = self.conn.add_new_transaction_query(tr_type, category, balance, description, date)
            self.reload_data()
            self.new_window.close()

    def optimize(self):
        opt_type = self.ui.comboBox.currentIndex()
        
        if opt_type == 1:
            goal = self.conn.get_goal()
            if goal is None:
                QtWidgets.QMessageBox.warning(self, "Ошибка", "цели нет")
                return
            g_amount = float(goal["g_amount"])
            raw_date = str(goal["g_date"]).strip()
            if len(raw_date) > 7:
                if raw_date[2] == '.':
                    raw_date = raw_date[3:]
                else:
                    raw_date = raw_date[-7:]
            parsed = False
            for fmt in ("%m.%Y", "%m/%Y", "%m-%Y"):
                try:
                    goal_date = datetime.datetime.strptime(raw_date, fmt)
                    parsed = True
                    break
                except Exception:
                    continue
            if not parsed:
                return
            current_month = getlmt(self.conn)
            dt_current = datetime.datetime.strptime(current_month, "%m.%Y")
            remaining_months = (goal_date.year - dt_current.year) * 12 + (goal_date.month - dt_current.month) + 1
            if remaining_months <= 0:
                QtWidgets.QMessageBox.warning(self, "Ошибка", "Дата прошла")
                return
            target_reduction = g_amount / remaining_months
            print(f"Тип оптимизации 'Выбранная цель': g_amount = {g_amount}, оставшихся месяцев = {remaining_months}, target_reduction = {target_reduction:.2f}")
        else:
            try:
                target_reduction = float(self.ui.opt_sum_line.text())
            except ValueError:
                QtWidgets.QMessageBox.warning(self, "ошибка", "неверно введена сумма")
                return

        month_query = getlmt(self.conn)
        if month_query is None:
            QtWidgets.QMessageBox.warning(self, "ошибка", "транз. нет")
            return

        income, outcomes, expense_categories = getinoutm(self.conn, month_query)
        x_raw = np.array([[income] + outcomes], dtype=np.float32)
        mean_tf = tf.constant(scaler.mean_, dtype=tf.float32)
        scale_tf = tf.constant(scaler.scale_, dtype=tf.float32)
        with tf.GradientTape() as tape:
            x_raw_tf = tf.convert_to_tensor(x_raw, dtype=tf.float32)
            tape.watch(x_raw_tf)
            x_scaled_tf = (x_raw_tf - mean_tf) / scale_tf
            predictions = model(x_scaled_tf)
        grads = tape.gradient(predictions, x_raw_tf)
        if grads is None:
            return
        grads = grads.numpy()[0]
        expense_indices = list(range(1, len(expense_categories)+1))
        scaling_factors = scaler.scale_[expense_indices]
        adjusted_grads = grads[expense_indices] / scaling_factors
        efficiency_list = []
        for i, idx in enumerate(expense_indices):
            cat = expense_categories[i]
            current_expense = outcomes[i]
            eff = -adjusted_grads[i]
            efficiency_list.append((cat, eff, current_expense))
        efficiency_list.sort(key=lambda x: x[1], reverse=True)
        prio_limits = {"Низкий": 0.70, "Средний": 0.40, "Высокий": 0.10}
        ui_priority = {
            "Еда": self.ui.food_prio.currentText(),
            "Здоровье и фитнес": self.ui.health_prio.currentText(),
            "Ком.платежи и сборы": self.ui.bills_prio.currentText(),
            "Подписки": self.ui.subs_prio.currentText(),
            "Покупки": self.ui.shop_prio.currentText(),
            "Транспорт": self.ui.transp_prio.currentText(),
            "Путешествия": self.ui.travs_prio.currentText(),
            "Развлечения": self.ui.enj_prio.currentText()
        }
        exclude_category = None
        if opt_type >= 2:
            mapping = {
                2: "Ком.платежи и сборы",
                3: "Транспорт",
                4: "Еда",
                5: "Здоровье и фитнес",
                6: "Подписки",
                7: "Покупки",
                8: "Транспорт",
                9: "Путешествия"
            }
            exclude_category = mapping.get(opt_type, None)
        
        remaining = target_reduction
        recommendations_dict = {cat: 0.0 for cat in expense_categories}
        for cat, eff, current in efficiency_list:
            if remaining <= 0:
                break
            if exclude_category is not None and cat == exclude_category:
                allowed = 0
            else:
                limit_str = ui_priority.get(cat, "Средний")
                limit_percentage = prio_limits.get(limit_str, 0.40)
                allowed = limit_percentage * current
            max_possible = min(allowed, remaining)
            if max_possible > 0:
                recommendations_dict[cat] = max_possible
                remaining -= max_possible

        display_order = [
            "Ком.платежи и сборы",
            "Путешествия",
            "Здоровье и фитнес",
            "Транспорт",
            "Покупки",
            "Подписки",
            "Еда",
            "Развлечения"
        ]
        self.ui.tableWidget.setRowCount(1)
        self.ui.tableWidget.setColumnCount(len(display_order))
        for col_index, category in enumerate(display_order):
            recommended_cut = recommendations_dict.get(category, 0)
            try:
                idx = expense_categories.index(category)
                current_expense = outcomes[idx]
            except ValueError:
                current_expense = 0
            if current_expense > 0:
                percent_cut = (recommended_cut / current_expense) * 100
            else:
                percent_cut = 0
            cell_text = f"{percent_cut:.2f}% ({recommended_cut:.2f} руб.)"
            item = QtWidgets.QTableWidgetItem(cell_text)
            self.ui.tableWidget.setItem(0, col_index, item)

    def opt_check_w(self):
        self.feedback_window = QtWidgets.QDialog()
        self.opt_ui = Ui_opt_check()
        self.opt_ui.setupUi(self.feedback_window)
        self.feedback_window.show()
        self.opt_ui.opt_good.clicked.connect(lambda: self.oper_fb(True))
        self.opt_ui.opt_bad.clicked.connect(lambda: self.oper_fb(False))
    
    def oper_fb(self, positive):
        current_date = datetime.datetime.now().strftime("%Y%m%d")
        recommendation = "Оптимизация"
        user_rating = 5 if positive else 1
        comment = ""
        if hasattr(self.opt_ui, "opt_comment"):
            comment = self.opt_ui.opt_comment.text()
        if self.conn.save_feedback(recommendation, user_rating, comment, current_date):
            print("Обратная связь сохранена.")
        self.reload_data()
        self.feedback_window.close()

    def prepftd(self, month_str):
        income, outcomes, expense_categories = getinoutm(self.conn, month_str)
        X = np.array([[income] + outcomes], dtype=np.float32)
        base_y = income - sum(outcomes)
        y = np.array([base_y], dtype=np.float32)
        fb_l = self.conn.get_all_feedback()
        dt = datetime.datetime.strptime(month_str, "%m.%Y")
        fb_m = dt.strftime("%Y%m")
        ratings = []
        for fb in fb_l:
            if str(fb["Date"]).startswith(fb_m):
                ratings.append(fb["user_rating"])
        if ratings:
            avg_rating = sum(ratings) / len(ratings)
            sample_weight = 1 + (5 - avg_rating)
        else:
            sample_weight = 1.0
        sw = np.array([sample_weight], dtype=np.float32)
        return X, y, sw

    def check_and_finetune(self):
        global ltm
        new_month = getlmt(self.conn)
        if new_month is None:
            return
        try:
            dt_new = datetime.datetime.strptime(new_month, "%m.%Y")
            dt_last = datetime.datetime.strptime(ltm, "%m.%Y")
        except Exception as e:
            print("проблема с датами:", e)
            return
        if dt_new > dt_last:
            training_month = get_previous_month(new_month)
            print(f"Дообучаем на {training_month}.")
            X, y, sw = self.prepftd(training_month)
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                          loss='mean_squared_error')
            history = model.fit(X, y, sample_weight=sw, epochs=5, verbose=1)
            ltm = new_month

    def reload_data_and_check(self):
        self.reload_data()
        self.check_and_finetune()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = finlogic()
    window.show()
    sys.exit(app.exec())
