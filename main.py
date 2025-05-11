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
from sklearn.model_selection import train_test_split
import xgboost as xgb

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
    class_weight='balanced',  #учет дисбаланса классов
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

# --- 7. Функция для вывода метрик ---
def print_metrics(model_name, y_train, y_train_pred, y_test, y_test_pred):
    print(f"{'-'*40}")
    print(f"{model_name} - Train Metrics")
    print(classification_report(y_train, y_train_pred))

    print(f"{model_name} - Test Metrics")
    print(classification_report(y_test, y_test_pred))
    print(f"{'-'*40}\n")

    cm_train = confusion_matrix(y_train, y_train_pred)
    disp_train = ConfusionMatrixDisplay(confusion_matrix=cm_train)
    disp_train.plot()
    plt.title(f"{model_name} - Confusion Matrix (Train)")
    plt.show()

    cm_test = confusion_matrix(y_test, y_test_pred)
    disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test)
    disp_test.plot()
    plt.title(f"{model_name} - Confusion Matrix (Test)")
    plt.show()

# --- 8. Вывод метрик ---
print_metrics("Random Forest", y_train, y_train_pred_rf, y_test, y_test_pred_rf)
print_metrics("XGBoost", y_train, y_train_pred_xgb, y_test, y_test_pred_xgb)

# --- 9. Feature Importance (Random Forest) ---
plt.figure(figsize=(10, 6))
sns.barplot(x=rf.feature_importances_, y=X_train.columns)
plt.title("Feature Importances - Random Forest")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

plot_importance(xgb_clf, max_num_features=10, height=0.5, title="Top 10 Features - XGBoost")
plt.show()