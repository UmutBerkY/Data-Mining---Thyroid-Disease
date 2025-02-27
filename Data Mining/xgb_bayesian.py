import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import optuna
import joblib  # joblib modülünü import etmelisiniz
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

# SMOTE uygulama
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Optuna ile Bayesian Optimization için objective fonksiyonu tanımladık.
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
        'subsample': trial.suggest_float('subsample', 0.3, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        'use_label_encoder': False,
        'eval_metric': 'logloss'
    }

    xgb = XGBClassifier(**params, random_state=42)
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Optuna çalıştırma
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# En iyi parametreler
best_params = study.best_params
print("Optuna ile bulunan en iyi parametreler:", best_params)

# En iyi parametrelerle model oluşturma ve eğitme
best_xgb = XGBClassifier(**best_params, random_state=42, use_label_encoder=False, eval_metric='logloss')
best_xgb.fit(X_train, y_train)

# Test setinde tahmin yapma
y_pred = best_xgb.predict(X_test)

# Performans raporu
print(classification_report(y_test, y_pred))

# Modeli kaydetme
joblib.dump(best_xgb, 'best_xgb_model.pkl')
print("Model 'best_xgb_bayesian_model.pkl' olarak kaydedildi.")

#Optuna ile bulunan en iyi parametreler: {'n_estimators': 279, 'max_depth': 7, 'learning_rate': 0.17583886661394266,

