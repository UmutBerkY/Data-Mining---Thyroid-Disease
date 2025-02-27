import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter
import joblib  # joblib modülünü import etmelisiniz
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

# Özellikler (X) ve hedef değişken (y)
X = combined_df.drop(columns=['Class'])  # "Class" dışındaki sütunlar
y = combined_df['Class']

# Sınıf dağılımını kontrol edin
print("SMOTE öncesi sınıf dağılımı:", Counter(y))

# SMOTE uygulama
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Yeni sınıf dağılımını kontrol edin
print("SMOTE sonrası sınıf dağılımı:", Counter(y_resampled))

# Dengelenmiş veriyi birleştirerek yeni bir DataFrame oluşturma
X_resampled_df = pd.DataFrame(X_resampled, columns=X.columns)  # Özellikler
y_resampled_df = pd.DataFrame(y_resampled, columns=['Class'])  # Hedef sütun

# X ve y'yi birleştirme
balanced_df = pd.concat([X_resampled_df, y_resampled_df], axis=1)

# Dengelenmiş veri setini kaydetme
balanced_df.to_csv('combined_dataset.csv', index=False)




