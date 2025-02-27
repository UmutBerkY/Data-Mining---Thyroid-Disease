import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

#Bu ayarı kodun başına ekleyerek tüm replace işlemleri için varsayılan davranışı korumuş oluruz.
pd.set_option('future.no_silent_downcasting', True)

combined_df=pd.read_csv("combined_dataset.csv")

# 7. Özellikleri (X) ve hedef değişkeni (y) ayırma
X = combined_df.drop(columns=['Class'])
y = combined_df['Class']

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. GridSearchCV ile LightGBM Modeli
# GridSearchCV kullanarak parametreleri optimize edelim

param_grid = {
    'objective': ['binary'],  # İkili sınıflandırma
    'metric': ['binary_error'],  # Hata oranı metriği
    'boosting_type': ['gbdt', 'dart'],  # Boosting türü (GBDT ya da DART)
    'num_leaves': [31, 50, 100],  # Ağaç yapısının derinliği
    'learning_rate': [0.01, 0.05, 0.1],  # Öğrenme oranı
    'feature_fraction': [0.7, 0.8, 0.9],  # Özelliklerin rastgele seçilmesi
    'bagging_fraction': [0.8, 0.9, 1.0],  # Örneklerin rastgele seçilmesi
    'bagging_freq': [1, 5],  # Bagging sıklığı
    'scale_pos_weight': [1, 5, 10],  # Pozitif sınıf ağırlığı
}

# Modelin oluşturulması
lgb_model = lgb.LGBMClassifier(random_state=42)

# GridSearchCV ile parametre arama
grid_search = GridSearchCV(estimator=lgb_model, param_grid=param_grid, cv=3, verbose=2, n_jobs=-1, scoring='accuracy')
grid_search.fit(X_train, y_train)

# En iyi parametreleri ve modeli al
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# En iyi parametreleri yazdırma
print("En iyi parametreler:", best_params)

# Test setinde tahmin yapma
y_pred = best_model.predict(X_test)

# Performans raporu
#print(classification_report(y_test, y_pred))

# Modeli kaydetme
joblib.dump(best_model, 'best_lgb_model.pkl')
print("Model 'best_lgb_model.pkl' olarak kaydedildi.")


#En iyi parametreler: {'bagging_fraction': 1.0, 'bagging_freq': 1, 'boosting_type': 'gbdt', 'feature_fraction': 0.7, 'learning_rate': 0.05,
#'metric': 'binary_error', 'num_leaves': 31, 'objective': 'binary', 'scale_pos_weight': 1}
