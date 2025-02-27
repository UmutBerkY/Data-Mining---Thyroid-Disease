import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

#Bu ayarı kodun başına ekleyerek tüm replace işlemleri için varsayılan davranışı korumuş oluruz.
pd.set_option('future.no_silent_downcasting', True)

# 1. Datasetlerin yüklenmesi
df1 = pd.read_csv('bir.csv')
df2 = pd.read_csv('iki.csv')
df3 = pd.read_csv('uc.csv')

# Datasetleri aynı forma getiriyoruz.
df1 = df1.drop(columns=['patient_id'])
df1['Class'] = df1['Class'].apply(lambda x: False if '-' in str(x) else True).astype(bool)
df2['Class'] = df2['Class'].replace({'N': False, 'P': True}).astype(bool)
df3['Class'] = df3['Class'].replace({'negative': False, 'sick': True}).astype(bool)

# 4. Datasetleri birleştirme
combined_df = pd.concat([df1, df2, df3], ignore_index=True)

# Boşluk ve '?' işaretlerini NaN ile değiştir
combined_df.replace({' ': pd.NA, '?': pd.NA}, inplace=True)

# Sayısal sütunları manuel olarak belirle
columns_to_fill = ['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI', 'TBG']

# Bu sütunları sayısal değerlere dönüştür
for column in columns_to_fill:
    if column in combined_df.columns:
        combined_df[column] = pd.to_numeric(combined_df[column], errors='coerce')

# Bu sütunlar için NaN değerlerini medyan ile doldur
for column in columns_to_fill:
    if column in combined_df.columns:
        median_value = combined_df[column].median()
        combined_df[column] = combined_df[column].fillna(median_value)

# Age sütunu 100'den küçük olan satırları kaldırma
combined_df = combined_df[combined_df['age'] <= 100]

# Boolean dönüşüm için sütunlar
columns_to_boolean = [
    'sex','thyroxine', 'query_on_thyroxine', 'on_antithyroid_medication', 'sick',
    'pregnant', 'thyroid_surgery', 'I131_treatment', 'query_hypothyroid',
    'query_hyperthyroid', 'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych',
    'TSH_measured', 'T3_measured', 'TT4_measured', 'T4U_measured', 'FTI_measured',
    'TBG_measured'
]

# Boolean sütunları düzenleme
for column in columns_to_boolean:
    if column in combined_df.columns:
        combined_df[column] = combined_df[column].apply(lambda x: True if str(x).lower() in ['t', 'm', 'yes', '1'] else False)

# 6. Kategorik verileri sayısal verilere dönüştürme (Label Encoding)
label_encoder = LabelEncoder()

# Sütun etiketlerinin sayısal hale getirilmesi
combined_df['Class'] = label_encoder.fit_transform(combined_df['Class'])
combined_df['sex'] = label_encoder.fit_transform(combined_df['sex'])
combined_df['referral_source'] = label_encoder.fit_transform(combined_df['referral_source'])

combined_df = combined_df.drop_duplicates()

#NİTELİK AZALTMA

# Girdi ve hedefi ayır
X = combined_df[['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI', 'TBG']]
y = combined_df['Class']

# Veriyi böl
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model oluştur ve eğit
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Özellik önemini gör

importance = model.feature_importances_
for sutun, onem in zip(X.columns, importance):
    print(f"{sutun}: Önem = {onem:.2f}")

#age: Önem = 0.12   TSH: Önem = 0.19    T3: Önem = 0.23     TT4: Önem = 0.14    T4U: Önem = 0.14    FTI: Önem = 0.16    TBG: Önem = 0.02
#TBG Çıkarıyoruz.
combined_df = combined_df.drop(columns=['TBG'])

from scipy.stats import chi2_contingency

# Binary sütunlar
binary_columns = [
    'sex', 'thyroxine', 'query_on_thyroxine', 'on_antithyroid_medication',
    'sick', 'pregnant', 'thyroid_surgery', 'I131_treatment',
    'query_hypothyroid', 'query_hyperthyroid', 'lithium', 'goitre',
    'tumor', 'hypopituitary', 'psych', 'TSH_measured', 'T3_measured',
    'TT4_measured', 'T4U_measured', 'FTI_measured', 'TBG_measured'
]

# Chi-Square testi
anlamsiz_binary_sutunlar = []
for col in binary_columns:
    crosstab = pd.crosstab(combined_df[col], combined_df['Class'])
    chi2, p, dof, _ = chi2_contingency(crosstab)
    if p > 0.05:
        anlamsiz_binary_sutunlar.append(col)

print("Anlamsız binary sütunlar:", anlamsiz_binary_sutunlar)

# Düşük ilişkili binary sütunları çıkar
combined_df = combined_df.drop(columns=anlamsiz_binary_sutunlar)


#Anlamsız binary sütunlar: ['query_on_thyroxine', 'on_antithyroid_medication', 'sick', 'thyroid_surgery', 'I131_treatment', 'query_hypothyroid',
#'query_hyperthyroid', 'goitre', 'tumor', 'hypopituitary']

# Belirtilen sütunlar
columns_to_check = ['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']

# Aykırı değerleri çıkarma işlemi
for col in columns_to_check:
    # Çeyrekler arası mesafeyi (IQR) hesapla
    Q1 = combined_df[col].quantile(0.25)
    Q3 = combined_df[col].quantile(0.75)
    IQR = Q3 - Q1

    # Aykırı değerlerin alt ve üst sınırlarını hesapla
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Aykırı değerleri çıkar
    combined_df = combined_df[(combined_df[col] >= lower_bound) &
                              (combined_df[col] <= upper_bound)]

import matplotlib.pyplot as plt
import seaborn as sns

# Sayısal sütunların kutu grafiğini çizme
plt.figure(figsize=(10, 6))
sns.boxplot(data=combined_df[['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']])
plt.title("Sayısal Sütunlar İçin Kutu Grafiği")
plt.show()

combined_df[['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']].hist(bins=20, figsize=(12, 8))
plt.suptitle("Sayısal Sütunlar İçin Histogramlar")
plt.show()

# Korelasyon matrisi
correlation_matrix = combined_df[['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']].corr()

# Isı haritası
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Korelasyon Isı Haritası")
plt.show()

