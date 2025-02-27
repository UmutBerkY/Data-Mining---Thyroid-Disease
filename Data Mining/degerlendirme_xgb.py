import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, matthews_corrcoef, cohen_kappa_score, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

#Bu ayarı kodun başına ekleyerek tüm replace işlemleri için varsayılan davranışı korumuş oluruz.
pd.set_option('future.no_silent_downcasting', True)

# 1. Datasetlerin yüklenmesi
combined_df=pd.read_csv("combined_dataset.csv")

# 2. Özellikleri (X) ve hedef değişkeni (y) ayırma
X = combined_df.drop(columns=['Class'])
y = combined_df['Class']

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. XGBoost modelini eğitme (Optuna parametreleri kullanarak)
xgb_model = xgb.XGBClassifier(
    n_estimators=279,
    max_depth=7,
    learning_rate=0.17583886661394266,
    colsample_bytree=0.6043423018983898,
    subsample=0.9521434515307706,
    min_child_weight=2,
    gamma=1.8136558788547656,
    reg_alpha=0.6872595696775945,
    reg_lambda=8.493691102872324,
    random_state=42
)

# Modeli eğitme
xgb_model.fit(X_train, y_train)

# 4. Tahminler
y_pred = xgb_model.predict(X_test)
y_pred_prob = xgb_model.predict_proba(X_test)[:, 1]  # AUC-ROC için gerekli olan probabilistik tahmin

# 5. Değerlendirme Metrikleri
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_prob)
conf_matrix = confusion_matrix(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)
kappa = cohen_kappa_score(y_test, y_pred)

# Fowlkes-Mallows Index (FMI) hesaplama
fmi = np.sqrt(precision * recall)

# 6. Sonuçları Yazdırma
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"AUC-ROC: {roc_auc}")
print(f"Matthews Correlation Coefficient (MCC): {mcc}")
print(f"Cohen's Kappa: {kappa}")
print(f"Fowlkes-Mallows Index: {fmi}")

# 7. Modeli Kaydetme (isteğe bağlı)
joblib.dump(xgb_model, 'best_xgb_model_with_metrics.pkl')
print("Model 'best_xgb_model_with_metrics.pkl' olarak kaydedildi.")

train_accuracy_xgb = xgb_model.score(X_train, y_train)
test_accuracy_xgb = xgb_model.score(X_test, y_test)

print(f"XGBoost - Eğitim Doğruluğu: {train_accuracy_xgb}")
print(f"XGBoost - Test Doğruluğu: {test_accuracy_xgb}")

# 8. AUC-ROC Eğrisini Çizme
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

# ROC Eğrisini Çiz
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# 9. Karmaşıklık Matrisini Görselleştirme
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'],
            annot_kws={'size': 16})
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()