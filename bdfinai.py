from PyQt6 import QtSql

class bdmain:
    def __init__(self):
        self.create_connection()

    def create_connection(self):
        bd = QtSql.QSqlDatabase.addDatabase('QSQLITE')
        bd.setDatabaseName('transactions.db')
        
        if not bd.open():
            print("Ошибка подключения:", bd.lastError().text())
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
        return True

    def execute_qr(self, sql_query, q_args=None):
        if not QtSql.QSqlDatabase.database().isOpen():
            print("База данных не открыта!")
            return False
        
        query = QtSql.QSqlQuery()
        query.prepare(sql_query)
        
        if q_args is not None:
            for arg in q_args:
                query.addBindValue(arg)
        
        if not query.exec():
            print("SQL-ошибка:", query.lastError().text())
            return False
        
        return query

    def add_new_transaction_query(self, type, category, amnt, cmnt, date):
        sql_query = "INSERT INTO transactions(TrType, Category, Amnt, Cmnt, Date) VALUES (?, ?, ?, ?, ?)"
        self.execute_qr(sql_query, [type, category, amnt, cmnt, date])

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
        t_out =  self.sum_pm(column="Amnt", filter="Category", value="Транспорт")
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