from PyQt6 import QtSql
import sqlite3

class bdmain:
    def __init__(self):
        self.create_connection()

    def create_connection(self):
        bd = QtSql.QSqlDatabase.addDatabase('QSQLITE')
        bd.setDatabaseName('transactions.db')
        
        if not bd.open():
            print("ошибка", bd.lastError().text())
            return False
        
        query = QtSql.QSqlQuery()
        query.exec(
            """
            CREATE TABLE IF NOT EXISTS transactions (
                Id INTEGER PRIMARY KEY AUTOINCREMENT,
                TrType VARCHAR(6),
                Category VARCHAR(27),
                Amnt REAL,
                Cmnt VARCHAR(20),
                Date VARCHAR(8)
            )
            """
        )
        query.exec(
            """
            CREATE TABLE IF NOT EXISTS goal (
                id INTEGER PRIMARY KEY,
                name TEXT,
                g_date TEXT,
                g_amount REAL
            )
            """
        )
        query.exec(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id INTEGER,
                recommendation TEXT,
                user_rating INTEGER,
                comment TEXT,
                Date TEXT
            )
            """
        )
        query.exec(
            """
            CREATE TABLE IF NOT EXISTS monthly_summary (
                month TEXT PRIMARY KEY,
                othincomeline REAL,
                secincomeline REAL,
                mainincomeline REAL,
                othout_line REAL,
                travs_line REAL,
                subs_line REAL,
                shop_line REAL,
                food_line REAL,
                health_line REAL,
                transp_line REAL,
                bills_line REAL,
                enj_line REAL
            )
            """
        )
        return True

    def execute_qr(self, sql_query, q_args=None):
        if not QtSql.QSqlDatabase.database().isOpen():
            return False
        
        query = QtSql.QSqlQuery()
        query.prepare(sql_query)
        
        if q_args is not None:
            for arg in q_args:
                query.addBindValue(arg)
        
        if not query.exec():
            print("ошибка", query.lastError().text())
            return False
        
        return query

    def add_new_transaction_query(self, type, category, amnt, cmnt, date):
        sql_query = """
            INSERT INTO transactions(TrType, Category, Amnt, Cmnt, Date)
            VALUES (?, ?, ?, ?, ?)
        """
        query = self.execute_qr(sql_query, [type, category, amnt, cmnt, date])
        inserted_id = query.lastInsertId()
        if inserted_id == -1:
            fallback_query = self.execute_qr("SELECT last_insert_rowid()")
            if fallback_query and fallback_query.next():
                inserted_id = fallback_query.value(0)
        return inserted_id

    def delete_transaction(self, trans_id):
        sql_query = "DELETE FROM transactions WHERE Id = ?"
        self.execute_qr(sql_query, [trans_id])

    def sum_pm(self, column, filter=None, value=None):
        sql_query = f"SELECT SUM({column}) FROM transactions"
        query_values = []
        if filter and value:
            sql_query += f" WHERE {filter} = ?"
            query_values.append(value)
        query = self.execute_qr(sql_query, query_values)
        if not query:
            return '0'
        if query.next():
            return str(query.value(0))
        return '0'

    def total_balance(self):
        return self.sum_pm(column="Amnt")

    def total_income(self):
        xxxx = self.sum_pm(column="Amnt", filter="TrType", value="Income")
        return f"Суммарный Доход:          {xxxx} ₽"

    def total_outcome(self):
        xxx = self.sum_pm(column="Amnt", filter="TrType", value="Outcome")
        return f"Суммарный Расход:          {xxx} ₽"

    def total_bills(self):
        b_out = self.sum_pm(column="Amnt", filter="Category", value="Ком.платежи и сборы")
        return f"Потрачено:{b_out} ₽"

    def total_transport(self):
        t_out = self.sum_pm(column="Amnt", filter="Category", value="Транспорт")
        return f"Потрачено:{t_out} ₽"

    def total_health(self):
        h_out = self.sum_pm(column="Amnt", filter="Category", value="Здоровье и фитнес")
        return f"Потрачено:{h_out} ₽"

    def total_food(self):
        f_out = self.sum_pm(column="Amnt", filter="Category", value="Еда")
        return f"Потрачено:{f_out} ₽"

    def total_shop(self):
        sh_out = self.sum_pm(column="Amnt", filter="Category", value="Покупки")
        return f"Потрачено:{sh_out} ₽"

    def total_subs(self):
        s_out = self.sum_pm(column="Amnt", filter="Category", value="Подписки")
        return f"Потрачено:{s_out} ₽"

    def total_travel(self):
        trav_out = self.sum_pm(column="Amnt", filter="Category", value="Путешествия")
        return f"Потрачено:{trav_out} ₽"

    def total_enj(self):
        e_out = self.sum_pm(column="Amnt", filter="Category", value="Развлечения")
        return f"Потрачено:{e_out} ₽"

    def total_other(self):
        o_out = self.sum_pm(column="Amnt", filter="Category", value="Другое")
        return f"Потрачено:{o_out} ₽"

    def clear_database(self):
        sql_query = "DELETE FROM transactions"
        self.execute_qr(sql_query)

    def set_goal(self, name, g_date, g_amount):
        sql_query = """
            INSERT OR REPLACE INTO goal (id, name, g_date, g_amount)
            VALUES (1, ?, ?, ?)
        """
        query = self.execute_qr(sql_query, [name, g_date, g_amount])
        if not query:
            print("Ошибка")
            return False
        return True

    def get_goal(self):
        sql_query = "SELECT name, g_date, g_amount FROM goal WHERE id = 1"
        query = self.execute_qr(sql_query)
        if not query:
            return None
        if query.next():
            return {"name": query.value(0), "g_date": query.value(1), "g_amount": query.value(2)}
        return None

    def bdgoal_name(self):
        goal = self.get_goal()
        if goal is None:
            return "Цель: не задана"
        return f"Цель: {goal['name']}"

    def bdgoal_date(self):
        goal = self.get_goal()
        if goal is None:
            return "Дата: не задана"
        return f"Дата: {goal['g_date']}"

    def bdgoal_amount(self):
        goal = self.get_goal()
        if goal is None:
            return "Сумма: не задана"
        return f"Сумма: {goal['g_amount']} ₽"

    def get_all_transactions(self):
        transactions = []
        sql = "SELECT Id, TrType, Category, Amnt, Cmnt, Date FROM transactions"
        query = self.execute_qr(sql)
        if not query:
            return transactions
        while query.next():
            row = {
                "Id": query.value(0),
                "TrType": query.value(1),
                "Category": query.value(2),
                "Amnt": query.value(3),
                "Cmnt": query.value(4),
                "Date": query.value(5)
            }
            transactions.append(row)
        return transactions

    def get_all_feedback(self):
        feedback_list = []
        sql = "SELECT id, transaction_id, recommendation, user_rating, comment, Date FROM feedback"
        query = self.execute_qr(sql)
        if not query:
            return feedback_list
        while query.next():
            row = {
                "id": query.value(0),
                "transaction_id": query.value(1),
                "recommendation": query.value(2),
                "user_rating": query.value(3),
                "comment": query.value(4),
                "Date": query.value(5)
            }
            feedback_list.append(row)
        return feedback_list

    def collect_dataset(self):
        conn = sqlite3.connect('transactions.db')
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM transactions")
        transactions = [dict(row) for row in cursor.fetchall()]
        cursor.execute("SELECT * FROM feedback")
        feedback = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return transactions, feedback

    def get_monthly_category_sums(self, yyyymm):
        sql = """
            SELECT Category, SUM(Amnt)
            FROM transactions
            WHERE SUBSTR(Date, 1, 6) = ?
            GROUP BY Category
        """
        query = self.execute_qr(sql, [yyyymm])
        if not query:
            return []
        results = []
        while query.next():
            cat = query.value(0)
            amt = query.value(1)
            results.append((cat, amt))
        return results

    def save_feedback(self, recommendation, user_rating, comment, date):
        sql = "INSERT INTO feedback (recommendation, user_rating, comment, Date) VALUES (?, ?, ?, ?)"
        query = self.execute_qr(sql, [recommendation, user_rating, comment, date])
        if query:
            return True
        return False

    def save_monthly_summary(self, month, othin, secin, mainin, othout, travs, subs, shop, food, health, transp, bills, enj):
        sql = """
            INSERT OR REPLACE INTO monthly_summary
            (month, othincomeline, secincomeline, mainincomeline, othout_line, travs_line, subs_line, shop_line, food_line, health_line, transp_line, bills_line, enj_line)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        query = self.execute_qr(sql, [month, othin, secin, mainin, othout, travs, subs, shop, food, health, transp, bills, enj])
        if query:
            return True
        return False

    def get_monthly_summary(self, month):
        sql = """
            SELECT othincomeline, secincomeline, mainincomeline, othout_line, travs_line, subs_line, shop_line, food_line, health_line, transp_line, bills_line, enj_line
            FROM monthly_summary
            WHERE month = ?
        """
        query = self.execute_qr(sql, [month])
        if query and query.next():
            return {
                "othincomeline": query.value(0),
                "secincomeline": query.value(1),
                "mainincomeline": query.value(2),
                "othout_line": query.value(3),
                "travs_line": query.value(4),
                "subs_line": query.value(5),
                "shop_line": query.value(6),
                "food_line": query.value(7),
                "health_line": query.value(8),
                "transp_line": query.value(9),
                "bills_line": query.value(10),
                "enj_line": query.value(11)
            }
        else:
            return None
