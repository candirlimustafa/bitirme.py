import pandas as pd

def normalize_and_save(input_excel_path, output_excel_path):
    # Excel dosyasını oku
    df = pd.read_excel(input_excel_path)
    son_sutun_input = df.iloc[:, -1]
    # Yeni bir DataFrame oluştur
    normalized_df = pd.DataFrame()

    # Her sütun için normalize işlemi yap
    for column in df.columns[:-1]:  # Son sütunu hariç diğer sütunları döngüye al
        # Sayısal olmayan değerleri filtrele
        numeric_values = df[column].dropna()
        # Normalizasyon işlemi (örneğin 0 ile 1 arasına)
        max_value = numeric_values.max()
        min_value = numeric_values.min()
        normalized_values = (0.8 * ((numeric_values - min_value) / (max_value - min_value)) + 0.1)
        # Yeni sütun adı oluştur
        new_column_name = "Normalized " + column
        # Normalize edilmiş değerleri yeni DataFrame'e ekle
        normalized_df[new_column_name] = normalized_values

    # Orijinal "Soc" sütununu yeni DataFrame'e ekle
    normalized_df['Soc'] = son_sutun_input

    # Yeni DataFrame'i Excel dosyasına kaydet
    normalized_df.to_excel(output_excel_path, index=False)

# Excel dosyasının yolu
input_excel_file_path = "verikumesi.xlsx"

# Yeni dosyanın kaydedileceği dosya yolu
output_excel_file_path = "d_min_max.xlsx"

# Verileri normalize et ve yeni dosyaya kaydet
normalize_and_save(input_excel_file_path, output_excel_file_path)