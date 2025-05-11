import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

def initial_data_analysis(df_train, df_test):
    fig, axes = plt.subplots(3, 2, figsize=(14, 18))
    
    # 1. Распределение целевой переменной (Churn) в тренировочном наборе
    sns.countplot(data=df_train, x='Churn', ax=axes[0, 0])
    axes[0, 0].set_title('Churn Distribution in Train Data')
    
    # 2. Гистограмма Total day minutes
    sns.histplot(df_train['Total day minutes'], bins=30, kde=True, ax=axes[0, 1])
    axes[0, 1].set_title('Total Day Minutes Distribution')
    
    # 3. Boxplot Customer service calls
    sns.boxplot(data=df_train, y='Customer service calls', ax=axes[1, 0])
    axes[1, 0].set_title('Customer Service Calls Distribution')
    
    # 4. Корреляционная матрица
    numeric_cols = df_train.select_dtypes(include=[np.number]).columns
    corr = df_train[numeric_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=axes[1, 1])
    axes[1, 1].set_title('Correlation Matrix')
    
    # 5. Распределение International plan
    sns.countplot(data=df_train, x='International plan', ax=axes[2, 0])
    axes[2, 0].set_title('International Plan Distribution')
    
    # 6. Распределение Voice mail plan
    sns.countplot(data=df_train, x='Voice mail plan', ax=axes[2, 1])
    axes[2, 1].set_title('Voice Mail Plan Distribution')
    
    plt.tight_layout()
    plt.show()

# --- 1. Загрузка данных ---
df_train = pd.read_csv('./data/churn-bigml-80.csv')
df_test = pd.read_csv('./data/churn-bigml-20.csv')

# --- 2. Удаление избыточных признаков ---
df_train.drop(columns=['Total day charge', 'Total eve charge', 
                       'Total night charge', 'Total intl charge'], inplace=True)
df_test.drop(columns=['Total day charge', 'Total eve charge', 
                      'Total night charge', 'Total intl charge'], inplace=True)

initial_data_analysis(df_train, df_test)

# --- 3. Кодирование регионов и категориальных признаков ---
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

# Удаляем исходный столбец 'State'
df_train.drop('State', axis=1, inplace=True)
df_test.drop('State', axis=1, inplace=True)

# One-Hot Encoding для категориальных признаков
categorical_cols = ['Region', 'International plan', 'Voice mail plan']
df_train = pd.get_dummies(df_train, columns=categorical_cols, drop_first=True)
df_test = pd.get_dummies(df_test, columns=categorical_cols, drop_first=True)

# Убедимся, что в тестовых данных есть все колонки из тренировочных
for col in df_train.columns:
    if col not in df_test.columns and col != 'Churn':
        df_test[col] = 0

# --- 4. Добавление новых признаков ---
def add_features(df):
    # Общее время разговоров
    df['Total minutes'] = df['Total day minutes'] + df['Total eve minutes'] + df['Total night minutes']
    # Средняя длительность звонка
    df['Avg call duration'] = df['Total minutes'] / (df['Total day calls'] + df['Total eve calls'] + df['Total night calls'])
    return df

df_train = add_features(df_train)
df_test = add_features(df_test)

# --- 5. Разделение на признаки и целевую переменную ---
X_train = df_train.drop(columns=['Churn'])
y_train = df_train['Churn']
X_test = df_test.drop(columns=['Churn'])
y_test = df_test['Churn']

# --- 6. Балансировка классов с SMOTE ---
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)

# --- 7. Обучение моделей с оптимизацией гиперпараметров ---
# Random Forest с GridSearch
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [2, 5]
}

rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, scoring='f1', cv=5)
rf_grid.fit(X_res, y_res)

# XGBoost с GridSearch
xgb_params = {
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'scale_pos_weight': [1, 5]
}

xgb_grid = GridSearchCV(XGBClassifier(objective='binary:logistic', eval_metric='logloss'), xgb_params, scoring='f1', cv=5)
xgb_grid.fit(X_res, y_res)

# --- 8. Предсказания на train и test ---
# Random Forest
y_train_pred_rf = rf_grid.best_estimator_.predict(X_train)
y_test_pred_rf = rf_grid.best_estimator_.predict(X_test)

# XGBoost
y_train_pred_xgb = xgb_grid.best_estimator_.predict(X_train)
y_test_pred_xgb = xgb_grid.best_estimator_.predict(X_test)

# --- 9. Вывод метрик ---
def print_metrics(model_name, y_train, y_train_pred, y_test, y_test_pred):
    print(f"{'-'*40}")
    print(f"{model_name} - Train Metrics")
    print(classification_report(y_train, y_train_pred))
    print(f"{model_name} - Test Metrics")
    print(classification_report(y_test, y_test_pred))
    print(f"{'-'*40}\n")

print_metrics("Random Forest", y_train, y_train_pred_rf, y_test, y_test_pred_rf)
print_metrics("XGBoost", y_train, y_train_pred_xgb, y_test, y_test_pred_xgb)

# --- 10. Матрицы ошибок ---
def plot_confusion_matrices():
    cm_train_rf = confusion_matrix(y_train, y_train_pred_rf)
    cm_test_rf = confusion_matrix(y_test, y_test_pred_rf)
    cm_train_xgb = confusion_matrix(y_train, y_train_pred_xgb)
    cm_test_xgb = confusion_matrix(y_test, y_test_pred_xgb)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ConfusionMatrixDisplay(cm_train_rf).plot(ax=axes[0], cmap='Blues', colorbar=False)
    axes[0].set_title('Random Forest - Train')
    ConfusionMatrixDisplay(cm_train_xgb).plot(ax=axes[1], cmap='Blues')
    axes[1].set_title('XGBoost - Train')
    plt.suptitle('Confusion Matrices on Training Data')
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ConfusionMatrixDisplay(cm_test_rf).plot(ax=axes[0], cmap='Blues', colorbar=False)
    axes[0].set_title('Random Forest - Test')
    ConfusionMatrixDisplay(cm_test_xgb).plot(ax=axes[1], cmap='Blues')
    axes[1].set_title('XGBoost - Test')
    plt.suptitle('Confusion Matrices on Test Data')
    plt.tight_layout()
    plt.show()

plot_confusion_matrices()

# --- 11. Feature Importance ---
def plot_feature_importances():
    rf_importances = pd.Series(rf_grid.best_estimator_.feature_importances_, index=X_train.columns)
    xgb_importances = pd.Series(xgb_grid.best_estimator_.feature_importances_, index=X_train.columns)

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

plot_feature_importances()