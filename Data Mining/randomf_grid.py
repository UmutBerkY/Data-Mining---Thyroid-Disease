import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

combined_df=pd.read_csv("combined_dataset.csv")

# 7. Özellikleri (X) ve hedef değişkeni (y) ayırma
X = combined_df.drop(columns=['Class'])
y = combined_df['Class']

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Hiperparametre optimizasyonu için Grid Search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

# En iyi parametrelerle model eğitimi
best_model = grid_search.best_estimator_
print("En iyi parametreler:", grid_search.best_params_)

# Test setinde tahmin yapma
y_pred = best_model.predict(X_test)

# Performans raporu
#print(classification_report(y_test, y_pred))

# Son modeli kaydetme (isteğe bağlı)
joblib.dump(best_model, 'best_random_forest_model.pkl')
print("Model 'best_random_forest_model.pkl' olarak kaydedildi.")

#En iyi parametreler: {'max_depth': 10, 'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 200}

