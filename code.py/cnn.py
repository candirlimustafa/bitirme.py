import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv1D, Flatten, Dense, MaxPooling1D # type: ignore

# Step 1: Load the Data
data = pd.read_excel('sigmoid.xlsx')
print(data.columns)

# Step 2: Prepare the Data
X = data.iloc[:, :3].values
y = data.iloc[:, 3:].values

# Step 3: Split Data into Train and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

# Reshape input data for CNN [samples, timesteps, features]
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Step 4: Build the CNN Model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Step 5: Train the Model
model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=1)

# Step 6: Evaluate the Model
y_pred = model.predict(X_test)

# Hata metriklerinin hesaplanması
r_squared = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("R-squared value:", r_squared)
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
    
    # CNN modelini oluştur ve eğit
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # Verilerin reshape edilmesi
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Modeli eğit
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)
    
    # Test seti üzerinde tahmin yap
    y_pred = model.predict(X_test)
    
    # Hata metriklerini hesapla
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    r2_values.append(r2)
    mae_values.append(mae)
    rmse_values.append(rmse)
    
    print(f"Trial {i+1}: R-squared: {r2}")
    print(f"Trial {i+1}: MAE: {mae}")
    print(f"Trial {i+1}: RMSE: {rmse}")

# Ortalama R-kareyi, MAE'yi ve RMSE'yi hesapla
mean_r2 = np.mean(r2_values)
mean_mae = np.mean(mae_values)
mean_rmse = np.mean(rmse_values)

print(f"Ortalama R-squared: {mean_r2}")
print(f"Ortalama MAE: {mean_mae}")
print(f"Ortalama RMSE: {mean_rmse}")
'''
