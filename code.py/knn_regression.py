import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

# Veri kümesini yükle
data = pd.read_excel('sigmoid.xlsx')

# Giriş ve çıkış verilerini ayır
''' medyan, minmax
X = data[['Feature_0', 'Feature_1', 'Feature_2']].values
y = data['Feature_3'].values
'''
''' VeriKumesi
X = data[['Battery Voltage [V]', 'Battery Current [A]', 'Battery Temperature [°C]']].values
y = data['SoC [%]'].values
'''
'''
X = data[['Normalized Battery Voltage [V]', 'Normalized Battery Current [A]', 'Normalized Battery Temperature [°C]']].values
y = data['Soc'].values
'''

X = data.iloc[:, :3].values
y = data.iloc[:, 3:].values


# Veri kümesini eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

# KNN regresyon modelini oluştur ve eğit
k = 9  # K değeri
knn_reg = KNeighborsRegressor(n_neighbors=k)
knn_reg.fit(X_train, y_train)

# Test setini kullanarak modeli değerlendir
y_pred = knn_reg.predict(X_test)

# Hata metriklerini hesapla
r_squared = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("R-squared:", r_squared)
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
    
    # KNN regresyon modelini oluştur ve eğit
    k = 9  # K değeri
    knn_reg = KNeighborsRegressor(n_neighbors=k)
    knn_reg.fit(X_train, y_train)
    
    # Test seti üzerinde tahmin yap
    y_pred = knn_reg.predict(X_test)
    
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
