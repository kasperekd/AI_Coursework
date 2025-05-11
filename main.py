import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from xgboost import XGBClassifier, plot_importance
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Установка стиля через seaborn
sns.set(style="whitegrid")
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 12,
    'figure.titlesize': 16
})

# --- 1. Загрузка данных ---
df_train = pd.read_csv('./data/churn-bigml-80.csv')
df_test = pd.read_csv('./data/churn-bigml-20.csv')

# --- 2. Удаление избыточных признаков ---
df_train.drop(columns=['Total day charge', 'Total eve charge',
                       'Total night charge', 'Total intl charge'], inplace=True)
df_test.drop(columns=['Total day charge', 'Total eve charge',
                      'Total night charge', 'Total intl charge'], inplace=True)

# --- 3. Кодирование категориальных признаков ---
region_mapping = {
    # Northeast
    'CT': 'Northeast', 'ME': 'Northeast', 'MA': 'Northeast', 'NH': 'Northeast',
    'RI': 'Northeast', 'VT': 'Northeast', 'NJ': 'Northeast', 'NY': 'Northeast', 'PA': 'Northeast',

    # Midwest
    'IL': 'Midwest', 'IN': 'Midwest', 'IA': 'Midwest', 'KS': 'Midwest',
    'MI': 'Midwest', 'MN': 'Midwest', 'MO': 'Midwest', 'NE': 'Midwest',
    'ND': 'Midwest', 'OH': 'Midwest', 'SD': 'Midwest', 'WI': 'Midwest',

    # South
    'AL': 'South', 'AR': 'South', 'DE': 'South', 'FL': 'South', 'GA': 'South',
    'KY': 'South', 'LA': 'South', 'MD': 'South', 'MS': 'South', 'NC': 'South',
    'OK': 'South', 'SC': 'South', 'TN': 'South', 'TX': 'South', 'VA': 'South',
    'WV': 'South', 'DC': 'South',

    # West
    'AK': 'West', 'AZ': 'West', 'CA': 'West', 'CO': 'West', 'HI': 'West',
    'ID': 'West', 'MT': 'West', 'NV': 'West', 'NM': 'West', 'OR': 'West',
    'UT': 'West', 'WA': 'West', 'WY': 'West'
}

df_train['Region'] = df_train['State'].map(region_mapping)
df_test['Region'] = df_test['State'].map(region_mapping)

categorical_cols = df_train.select_dtypes(include=['object']).columns
label_encoder = LabelEncoder()

for col in categorical_cols:
    df_train[col] = label_encoder.fit_transform(df_train[col])
    df_test[col] = label_encoder.transform(df_test[col])

# --- 4. Разделение на признаки и целевую переменную ---
X_train = df_train.drop(columns=['Churn'])
y_train = df_train['Churn']
X_test = df_test.drop(columns=['Churn'])
y_test = df_test['Churn']

# --- 5. Обучение моделей ---
# Random Forest
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight='balanced',  
    random_state=42
)
rf.fit(X_train, y_train)

# XGBoost
neg_class = sum(y_train == 0)
pos_class = sum(y_train == 1)
scale_pos_weight = neg_class / pos_class

xgb_clf = XGBClassifier(
    objective='binary:logistic',
    scale_pos_weight=scale_pos_weight,
    eval_metric='logloss',
    max_depth=3,               
    learning_rate=0.1,         
    subsample=0.8,             
    colsample_bytree=0.8,      
    reg_lambda=1.0,            
    random_state=42
)
xgb_clf.fit(X_train, y_train)

# --- 6. Предсказания на train и test ---
# Random Forest
y_train_pred_rf = rf.predict(X_train)
y_test_pred_rf = rf.predict(X_test)

# XGBoost
y_train_pred_xgb = xgb_clf.predict(X_train)
y_test_pred_xgb = xgb_clf.predict(X_test)

# --- 7. Функция для вывода метрик без матриц ошибок ---
def print_metrics(model_name, y_train, y_train_pred, y_test, y_test_pred):
    print(f"{'-'*40}")
    print(f"{model_name} - Train Metrics")
    print(classification_report(y_train, y_train_pred))

    print(f"{model_name} - Test Metrics")
    print(classification_report(y_test, y_test_pred))
    print(f"{'-'*40}\n")

# --- 8. Вывод метрик ---
print_metrics("Random Forest", y_train, y_train_pred_rf, y_test, y_test_pred_rf)
print_metrics("XGBoost", y_train, y_train_pred_xgb, y_test, y_test_pred_xgb)

# --- 9. Построение объединенных матриц ошибок ---
cm_train_rf = confusion_matrix(y_train, y_train_pred_rf)
cm_test_rf = confusion_matrix(y_test, y_test_pred_rf)
cm_train_xgb = confusion_matrix(y_train, y_train_pred_xgb)
cm_test_xgb = confusion_matrix(y_test, y_test_pred_xgb)

# Матрицы ошибок на одном графике для train
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ConfusionMatrixDisplay(cm_train_rf).plot(ax=axes[0], cmap='Blues', colorbar=False)
axes[0].set_title('Random Forest - Train')
ConfusionMatrixDisplay(cm_train_xgb).plot(ax=axes[1], cmap='Blues')
axes[1].set_title('XGBoost - Train')
plt.suptitle('Confusion Matrices on Training Data')
plt.tight_layout()
plt.show()

# Матрицы ошибок на одном графике для test
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ConfusionMatrixDisplay(cm_test_rf).plot(ax=axes[0], cmap='Blues', colorbar=False)
axes[0].set_title('Random Forest - Test')
ConfusionMatrixDisplay(cm_test_xgb).plot(ax=axes[1], cmap='Blues')
axes[1].set_title('XGBoost - Test')
plt.suptitle('Confusion Matrices on Test Data')
plt.tight_layout()
plt.show()

# --- 10. Feature Importance ---
rf_importances = pd.Series(rf.feature_importances_, index=X_train.columns)
xgb_importances = pd.Series(xgb_clf.feature_importances_, index=X_train.columns)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.barplot(x=rf_importances.values, y=rf_importances.index, ax=axes[0])
axes[0].set_title("Feature Importances - Random Forest")
axes[0].set_xlabel("Importance")
axes[0].set_ylabel("Feature")

sns.barplot(x=xgb_importances.values, y=xgb_importances.index, ax=axes[1])
axes[1].set_title("Feature Importances - XGBoost")
axes[1].set_xlabel("Importance")
axes[1].set_ylabel("")
plt.suptitle('Feature Importances Comparison')
plt.tight_layout()
plt.show()