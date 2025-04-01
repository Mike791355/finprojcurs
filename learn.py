import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt


df = pd.read_csv('/content/transactions_course.csv', parse_dates=['Date']).sort_values('Date')


df['Month'] = df['Date'].dt.to_period('M')

monthly = df.pivot_table(index='Month',
                         columns=['TrType', 'Category'],
                         values='Amnt',
                         aggfunc='sum',
                         fill_value=0).reset_index().sort_values('Month')

monthly['Total_Income'] = monthly['доход'].sum(axis=1)
monthly['Total_Expense'] = monthly['расход'].sum(axis=1)
monthly['Savings'] = monthly['Total_Income'] - monthly['Total_Expense']

X = monthly.drop(['Month', 'Savings'], axis=1)
y = monthly['Savings']

feature_columns = X.columns

total_months = len(monthly)
train_size = int(total_months * 0.7)
val_size = int(total_months * 0.15)
test_size = total_months - train_size - val_size

X_train = X.iloc[:train_size]
X_val = X.iloc[train_size:train_size+val_size]
X_test = X.iloc[train_size+val_size:]

y_train = y.iloc[:train_size]
y_val = y.iloc[train_size:train_size+val_size]
y_test = y.iloc[train_size+val_size:]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)
])
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

early_stop = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)
history = model.fit(X_train_scaled, y_train,
                    validation_data=(X_val_scaled, y_val),
                    epochs=200,
                    batch_size=2,
                    callbacks=[early_stop],
                    verbose=1)

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training Progress on Main Dataset')
plt.ylabel('MSE')
plt.xlabel('Epoch')
plt.legend()
plt.show()

y_pred = model.predict(X_test_scaled)
mae = np.mean(np.abs(y_pred.flatten() - y_test.values))
r2 = r2_score(y_test, y_pred.flatten())
print(f'[TEST] MAE: {mae:.2f} руб.')
print(f'[TEST] R²: {r2:.2f}')

baseline_pred = np.full_like(y_test, y_train.mean())
baseline_mae = np.mean(np.abs(baseline_pred - y_test))
baseline_r2 = r2_score(y_test, baseline_pred)
print(f'\n[Baseline] MAE: {baseline_mae:.2f} руб.')
print(f'[Baseline] R²: {baseline_r2:.2f}')


new_files = [
    '/content/transactions_user1.csv',
    '/content/transactions_user2.csv',
    '/content/transactions_user4.csv'
]

def prepare_dataset(file, feature_columns):
    df_new = pd.read_csv(file, parse_dates=['Date']).sort_values('Date')
    df_new['Month'] = df_new['Date'].dt.to_period('M')
    monthly_new = df_new.pivot_table(index='Month',
                                     columns=['TrType', 'Category'],
                                     values='Amnt',
                                     aggfunc='sum',
                                     fill_value=0).reset_index().sort_values('Month')
    monthly_new['Total_Income'] = monthly_new['доход'].sum(axis=1)
    monthly_new['Total_Expense'] = monthly_new['расход'].sum(axis=1)
    monthly_new['Savings'] = monthly_new['Total_Income'] - monthly_new['Total_Expense']
    X_new = monthly_new.drop(['Month', 'Savings'], axis=1)
    y_new = monthly_new['Savings']
    X_new = X_new.reindex(columns=feature_columns, fill_value=0)
    return X_new, y_new

X_new_list = []
y_new_list = []

for file in new_files:
    X_new, y_new = prepare_dataset(file, feature_columns)
    X_new_list.append(X_new)
    y_new_list.append(y_new)


X_new_combined = pd.concat(X_new_list, ignore_index=True)
y_new_combined = pd.concat(y_new_list, ignore_index=True)


X_new_combined_scaled = scaler.transform(X_new_combined)


early_stop_ft = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history_ft = model.fit(X_new_combined_scaled, y_new_combined,
                       validation_split=0.15,
                       epochs=100,
                       batch_size=2,
                       callbacks=[early_stop_ft],
                       verbose=1)


plt.plot(history_ft.history['loss'], label='Fine-tune Train Loss')
plt.plot(history_ft.history['val_loss'], label='Fine-tune Val Loss')
plt.title('Fine-tuning Training Progress on Synthetic New Datasets')
plt.ylabel('MSE')
plt.xlabel('Epoch')
plt.legend()
plt.show()


y_new_pred = model.predict(X_new_combined_scaled)
mae_new = np.mean(np.abs(y_new_pred.flatten() - y_new_combined.values))
r2_new = r2_score(y_new_combined, y_new_pred.flatten())
print(f'\n[New Synthetic Data] MAE: {mae_new:.2f} руб.')
print(f'[New Synthetic Data] R²: {r2_new:.2f}')



real_files = [
    '/content/moutput6.csv',
    '/content/moutput8.csv'
]

X_real_list = []
y_real_list = []

for file in real_files:
    X_real, y_real = prepare_dataset(file, feature_columns)
    X_real_list.append(X_real)
    y_real_list.append(y_real)

X_real_combined = pd.concat(X_real_list, ignore_index=True)
y_real_combined = pd.concat(y_real_list, ignore_index=True)

X_real_combined_scaled = scaler.transform(X_real_combined)

early_stop_real = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)
history_real = model.fit(X_real_combined_scaled, y_real_combined,
                         validation_split=0.15,
                         epochs=100,
                         batch_size=2,
                         callbacks=[early_stop_real],
                         verbose=1)

plt.plot(history_real.history['loss'], label='Real Fine-tune Train Loss')
plt.plot(history_real.history['val_loss'], label='Real Fine-tune Val Loss')
plt.title('Fine-tuning Training Progress on Real Datasets')
plt.ylabel('MSE')
plt.xlabel('Epoch')
plt.legend()
plt.show()


y_real_pred = model.predict(X_real_combined_scaled)
mae_real = np.mean(np.abs(y_real_pred.flatten() - y_real_combined.values))
r2_real = r2_score(y_real_combined, y_real_pred.flatten())
print(f'\n[Real Data] MAE: {mae_real:.2f} руб.')
print(f'[Real Data] R²: {r2_real:.2f}')
