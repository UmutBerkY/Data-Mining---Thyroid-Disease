import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
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

# Bayesian Optimization için parametre aralığı
param_space = {
    'hidden_layer_sizes': Integer(50, 200),  # Gizli katman büyüklüğü
    'activation': ['relu', 'tanh'],  # Aktivasyon fonksiyonu
    'solver': ['adam', 'sgd'],  # Optimizasyon algoritması
    'alpha': Real(1e-5, 1e-1, prior='log-uniform'),  # L2 düzenlileştirme katsayısı
    'learning_rate': ['constant', 'adaptive'],  # Öğrenme hızı
    'max_iter': Integer(1000, 2000),  # Maksimum iterasyon sayısını artırmak
    'tol': Real(1e-4, 1e-2)  # Daha hassas bir tolerans ayarı
}

# MLP (yapay sinir ağı) modeli oluşturma
mlp = MLPClassifier(random_state=42)

# Bayesian Optimization uygulama
opt = BayesSearchCV(mlp, param_space, n_iter=50, cv=5, scoring='accuracy', n_jobs=-1, verbose=2, random_state=42)
opt.fit(X_train, y_train)

# En iyi parametreleri yazdırma
print("En iyi parametreler:", opt.best_params_)

# En iyi model ile tahmin yapma
best_model = opt.best_estimator_
y_pred = best_model.predict(X_test)

# Performans raporu
#print(classification_report(y_test, y_pred))

# Son modeli kaydetme
joblib.dump(best_model, 'best_mlp_bayesian_model.pkl')
print("Model 'best_mlp_bayesian_model.pkl' olarak kaydedildi.")


#En iyi parametreler: OrderedDict({'activation': 'tanh', 'alpha': 4.4722948143922705e-05, 'hidden_layer_sizes': 189, 'learning_rate': 'constant',

