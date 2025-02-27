import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from scipy.stats import randint
import joblib  # joblib modülünü import etmelisiniz

combined_df=pd.read_csv("combined_dataset.csv")

# 7. Özellikleri (X) ve hedef değişkeni (y) ayırma
X = combined_df.drop(columns=['Class'])
y = combined_df['Class']

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Search için parametre aralığı
param_dist = {
    'n_estimators': randint(50, 500),
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 20),
    'bootstrap': [True, False],
    'n_jobs': [-1]
}

# Random Forest modelini oluşturma
rf = RandomForestClassifier(random_state=42)

# RandomizedSearchCV uygulama
random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, n_iter=100, cv=5, scoring='accuracy', verbose=2, n_jobs=-1, random_state=42)
random_search.fit(X_train, y_train)

# En iyi parametreleri yazdırma
print("En iyi parametreler:", random_search.best_params_)

# En iyi model ile tahmin yapma
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)

# Performans raporu
print(classification_report(y_test, y_pred))

# Son modeli kaydetme
joblib.dump(best_model, 'best_random_forest_random_search_model.pkl')
print("Model 'best_random_forest_random_search_model.pkl' olarak kaydedildi.")


#En iyi parametreler: {'bootstrap': False, 'max_depth': 20, 'min_samples_leaf': 12, 'min_samples_split': 4, 'n_estimators': 356, 'n_jobs': -1}
#              precision    recall  f1-score   support
#
#           0       0.69      0.82      0.75      1537
#           1       0.77      0.63      0.70      1527
#
#    accuracy                           0.72      3064
#   macro avg       0.73      0.72      0.72      3064
#weighted avg       0.73      0.72      0.72      3064