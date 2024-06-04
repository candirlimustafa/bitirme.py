import pandas as pd
import numpy as np

def read_data(filename):
    data = pd.read_excel(filename)  # Excel dosyasını oku
    X = data.iloc[:, :-1].values    # Özellikler (son sütunu hariç)
    y = data.iloc[:, -1].values     # Hedef değişken (son sütun)
    return X, y

def median_normalization(X):
    medians = np.median(X, axis=0)  # Her özelliğin medyanını hesapla
    std_devs = np.std(X, axis=0)    # Her özelliğin standart sapmasını hesapla
    X_normalized = (X - medians) / std_devs  # Medyan normalizasyonu uygula
    return X_normalized

# Ana kod
dosya_adi = "VeriKumesi.xlsx"  # Veri dosyanızın adını buraya yazın
X, y = read_data(dosya_adi)

# Medyan normalizasyonu uygula
X_normalized = median_normalization(X)

# Veriyi parçalara böl
chunk_size = 100000  # Her parça için maksimum satır sayısı
num_chunks = len(X_normalized) // chunk_size + 1

for i in range(num_chunks):
    start_idx = i * chunk_size
    end_idx = min((i + 1) * chunk_size, len(X_normalized))
    chunk = X_normalized[start_idx:end_idx]
    
    # Normalize edilmiş veriyi pandas DataFrame'e dönüştür
    df_chunk = pd.DataFrame(chunk, columns=[f"Feature_{i}" for i in range(chunk.shape[1])])
    
    # Yeni Excel dosyasına yaz
    output_filename = f"NormalizeEdilmisVeri_Part_{i + 1}.xlsx"
    df_chunk.to_excel(output_filename, index=False)
    print(f"Normalize edilmiş veri '{output_filename}' dosyasına yazıldı.")

'''
import pandas as pd
import numpy as np

def read_data(filename):
    data = pd.read_excel(filename)  # Excel dosyasını oku
    X = data.iloc[:, :].values    # Özellikler (son sütunu hariç)
    y = data.iloc[:, -1].values     # Hedef değişken (son sütun)
    return X, y

def median_normalization(X):
    medians = np.median(X, axis=0)  # Her özelliğin medyanını hesapla
    std_devs = np.std(X, axis=0)    # Her özelliğin standart sapmasını hesapla
    X_normalized = (X - medians) / std_devs  # Medyan normalizasyonu uygula
    return X_normalized

# Ana kod
dosya_adi = "VeriKumesi.xlsx"  # Veri dosyanızın adını buraya yazın
X, y = read_data(dosya_adi)

# Medyan normalizasyonu uygula
X_normalized = median_normalization(X)
X_normalized[:, 3] = X[:, -1]

# Veriyi parçalara böl
chunk_size = 1000000  # Her parça için maksimum satır sayısı
num_chunks = 1 #len(X_normalized) // chunk_size + 1

for i in range(num_chunks):
    start_idx = i * chunk_size
    end_idx = min((i + 1) * chunk_size, len(X_normalized))
    chunk = X_normalized[start_idx:end_idx]

    # Normalize edilmiş veriyi pandas DataFrame'e dönüştür
    df_chunk = pd.DataFrame(chunk, columns=[f"Feature_{i}" for i in range(chunk.shape[1])])

    # Yeni Excel dosyasına yaz
    output_filename = f"medyan.xlsx"
    df_chunk.to_excel(output_filename, index=False)
    print(f"Normalize edilmiş veri '{output_filename}' dosyasına yazıldı.")

print('min0')
print(min(X_normalized[:, 0]))
print('max0')
print(max(X_normalized[:, 0]))
print('min1')
print(min(X_normalized[:, 1]))
print('max1')
print(max(X_normalized[:, 1]))
print('min2')
print(min(X_normalized[:, 2]))
print('max2')
print(max(X_normalized[:, 2]))
print('min3')
print(min(X_normalized[:, 3]))
print('max3')
print(max(X_normalized[:, 3]))
'''