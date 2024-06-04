import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def read_data(filename):
    data = pd.read_excel(filename)  # Excel dosyasını oku
    X = data.iloc[:, :-1].values    # Özellikler (son sütunu hariç)
    y = data.iloc[:, -1].values     # Hedef değişken (son sütun)
    return X, y

def fit_linear_regression(X_train, y_train):
    # Bias terimini ekleyelim
    X_train = np.column_stack((np.ones(len(X_train)), X_train))
    # Katsayıları hesaplayalım
    beta = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(y_train)
    return beta

def predict(X_test, beta):
    # Bias terimini ekleyelim
    X_test = np.column_stack((np.ones(len(X_test)), X_test))
    # Tahminler yapalım
    y_pred = X_test.dot(beta)
    return y_pred

def r_squared(y_true, y_pred):
    SSR = np.sum((y_pred - y_true) ** 2)  # Kareler toplamı
    SST = np.sum((y_true - np.mean(y_true)) ** 2)  # Toplam kareler toplamı
    r2 = 1 - (SSR / SST)
    return r2

def mean_absolute_error(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    return mae

def root_mean_squared_error(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return rmse

# Ana kod
dosya_adi = "d_min_max.xlsx"  # Veri dosyanızın adını buraya yazın
X, y = read_data(dosya_adi)

# Modelin performansını 10 kez ölçelim
num_trials = 1
r2_values = []
mae_values = []
rmse_values = []

for i in range(num_trials):
    # Veriyi rastgele eğitim ve test setlerine ayır
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

    # Modeli eğit
    beta = fit_linear_regression(X_train, y_train)

    # Test seti üzerinde tahmin yap
    y_pred = predict(X_test, beta)

    # R-kareyi hesapla ve listeye ekle
    r2 = r_squared(y_test, y_pred)
    r2_values.append(r2)

    # MAE'yi hesapla ve listeye ekle
    mae = mean_absolute_error(y_test, y_pred)
    mae_values.append(mae)

    # RMSE'yi hesapla ve listeye ekle
    rmse = root_mean_squared_error(y_test, y_pred)
    rmse_values.append(rmse)

    print(f"Trial {i+1}: R-kare hatası: {r2}")
    print(f"Trial {i+1}: MAE: {mae}")
    print(f"Trial {i+1}: RMSE: {rmse}")

# Ortalama R-kareyi hesapla
mean_r2 = np.mean(r2_values)
mean_mae = np.mean(mae_values)
mean_rmse = np.mean(rmse_values)

print(f"Ortalama R-kare: {mean_r2}")
print(f"Ortalama MAE: {mean_mae}")
print(f"Ortalama RMSE: {mean_rmse}")
