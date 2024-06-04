import pandas as pd
import numpy as np
from keras.layers import Dense, LSTM, Bidirectional
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Excel dosyasını yükleme
excel_file = 'minmax.xlsx'
data = pd.read_excel(excel_file)

# Verileri DataFrame'e dönüştürme
veriler = pd.DataFrame(data)

# Veri setini eğitim, doğrulama ve test setlerine bölme
train_data, test_data = train_test_split(veriler, test_size=0.2, random_state=7)
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=7)

# BiLSTM modelinin oluşturulması
model = Sequential()
model.add(Bidirectional(LSTM(units=50, activation='relu'), input_shape=(3, 1)))
model.add(Dense(1))
model.compile(optimizer=Adam(), loss='mse')

# Verilerin reshape edilmesi
X_train = train_data.iloc[:, :-1].values.reshape((train_data.shape[0], 3, 1))
y_train = train_data.iloc[:, -1].values
X_val = val_data.iloc[:, :-1].values.reshape((val_data.shape[0], 3, 1))
y_val = val_data.iloc[:, -1].values
X_test = test_data.iloc[:, :-1].values.reshape((test_data.shape[0], 3, 1))
y_test = test_data.iloc[:, -1].values

# Modelin eğitilmesi
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val))

# Modelin değerlendirilmesi
evaluation = model.evaluate(X_val, y_val)

# Tahminlerin yapılması
predictions = model.predict(X_test)

# Hata metriklerinin hesaplanması
r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("R-kare Hatası:", r2)
print("MAE:", mae)
print("RMSE:", rmse)

'''
# Modelin performansını 5 kez ölçelim
num_trials = 5
r2_values = []
mae_values = []
rmse_values = []

for i in range(num_trials):
    # Veriyi rastgele eğitim ve test setlerine ayır
    train_data, test_data = train_test_split(veriler, test_size=0.2, random_state=i)
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=i)

    # BiLSTM modelinin oluşturulması
    model = Sequential()
    model.add(Bidirectional(LSTM(units=50, activation='relu'), input_shape=(3, 1)))
    model.add(Dense(1))
    model.compile(optimizer=Adam(), loss='mse')

    # Verilerin reshape edilmesi
    X_train = train_data.iloc[:, :-1].values.reshape((train_data.shape[0], 3, 1))
    y_train = train_data.iloc[:, -1].values
    X_val = val_data.iloc[:, :-1].values.reshape((val_data.shape[0], 3, 1))
    y_val = val_data.iloc[:, -1].values
    X_test = test_data.iloc[:, :-1].values.reshape((test_data.shape[0], 3, 1))
    y_test = test_data.iloc[:, -1].values

    # Modelin eğitilmesi
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val))

    # Tahminlerin yapılması
    predictions = model.predict(X_test)

    # Hata metriklerinin hesaplanması
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    r2_values.append(r2)
    mae_values.append(mae)
    rmse_values.append(rmse)

    print(f"Trial {i+1}: R-kare Hatası: {r2}")
    print(f"Trial {i+1}: MAE: {mae}")
    print(f"Trial {i+1}: RMSE: {rmse}")

# Ortalama R-kareyi, MAE'yi ve RMSE'yi hesapla
mean_r2 = np.mean(r2_values)
mean_mae = np.mean(mae_values)
mean_rmse = np.mean(rmse_values)

print(f"Ortalama R-kare: {mean_r2}")
print(f"Ortalama MAE: {mean_mae}")
print(f"Ortalama RMSE: {mean_rmse}")
'''
