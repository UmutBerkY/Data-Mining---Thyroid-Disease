import pandas as pd

# 1. Dataseti oku
df1 = pd.read_csv("bir.csv")  # Dosya adını doğru gir

# Sayısal olmayan değerleri NaN olarak işaretle
df1 = df1.apply(pd.to_numeric, errors='coerce')

# Korelasyon matrisi hesapla
correlation_matrix = df1.corr()

# patient_id sütununun diğer sütunlarla korelasyonunu al
if 'patient_id' in df1.columns:
    patient_id_correlation = correlation_matrix['patient_id']
    print("Patient ID sütununun diğer sütunlarla olan korelasyonu:")
    print(patient_id_correlation)

    # Korelasyon düşükse bunu belirt
    threshold = 0.1
    if patient_id_correlation.abs().drop('patient_id').max() < threshold:
        print(f"Patient ID sütunu diğer değişkenlerle düşük korelasyon (<{threshold}) gösteriyor. Bu nedenle çıkarılabilir.")
    else:
        print("Patient ID sütunu bazı değişkenlerle anlamlı korelasyon gösterebilir.")
else:
    print("Patient ID sütunu veri çerçevesinde bulunamadı.")