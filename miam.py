import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

df = pd.read_excel('2onexo.xlsx')
df = df.dropna(subset=['ASAK'])

for column in df.columns:
    if df[column].dtype in ['float64', 'int64']:
        df[column].fillna(df[column].median(), inplace=True)
    else:
        df[column].fillna(df[column].mode()[0], inplace=True)


plt.figure(figsize=(8, 6))
sns.histplot(df['ASAK'], kde=True)
plt.title('Распределение значений ASAK')
plt.xlabel('ASAK')
plt.ylabel('Частота')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x=df['ASAK'])
plt.title('Ящик с усами для ASAK')
plt.show()

Q1 = df['ASAK'].quantile(0.25)
Q3 = df['ASAK'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['ASAK'] >= Q1 - 1.5*IQR) & (df['ASAK'] <= Q3 + 1.5*IQR)]

numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
corr_matrix = df[numeric_cols].corr()

plt.figure(figsize=(15, 15))
sns.heatmap(corr_matrix, annot=True, fmt=".1f", cmap='coolwarm')
plt.title('Матрица корреляций')
plt.show()

corr_with_target = corr_matrix['ASAK'].sort_values(ascending=False)
print("Корреляция признаков с ASAK:")
print(corr_with_target)

selected_features = corr_with_target[abs(corr_with_target) > 0.3].index.tolist()
selected_features.remove('ASAK')


X = df[selected_features]
y = df['ASAK']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=52)
# Масштабирование признаков
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Модель: {model.__class__.__name__}")
    print(f"MSE: {mse:.4f}")
    print(f"R2: {r2:.4f}")
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Фактические значения')
    plt.ylabel('Предсказанные значения')
    plt.title(f'{model.__class__.__name__}: Факт vs Предсказание')
    plt.show()
    return model, mse, r2
print("\nОценка линейной регрессии:")
lr = LinearRegression()
lr_model, lr_mse, lr_r2 = evaluate_model(lr, X_train_scaled, X_test_scaled, y_train, y_test)
print("\nОценка случайного леса:")
rf = RandomForestRegressor(random_state=52)
rf_model, rf_mse, rf_r2 = evaluate_model(rf, X_train, X_test, y_train, y_test)
best_model = rf_model if rf_r2 > lr_r2 else lr_model
best_model_name = 'RandomForest' if rf_r2 > lr_r2 else 'LinearRegression'

if hasattr(best_model, 'feature_importances_'):
    feature_importance = best_model.feature_importances_
else:
    feature_importance = np.abs(best_model.coef_)
importance_df = pd.DataFrame({
    'Feature': selected_features,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)
importance_df['Importance_percent'] = (importance_df['Importance'] / importance_df['Importance'].sum()) * 100
print("\nВажность признаков в модели:")
print(importance_df)
important_features = importance_df[importance_df['Importance_percent'] >= 5]['Feature'].tolist()
print(f"\nПризнаки с важностью >=5%: {important_features}")
plt.figure(figsize=(12, 6))
sns.barplot(x='Importance_percent', y='Feature', data=importance_df)
plt.title('Важность признаков (%)')
plt.xlabel('Процент важности')
plt.ylabel('Признак')
plt.show()

original_df = pd.read_excel('2onexo.xlsx')

all_features = selected_features
df_for_pred = original_df.copy()
for column in df_for_pred.columns:
    if df_for_pred[column].dtype in ['float64', 'int64']:
        df_for_pred[column].fillna(df_for_pred[column].median(), inplace=True)
    else:
        df_for_pred[column].fillna(df_for_pred[column].mode()[0], inplace=True)
for feature in all_features:
    if feature not in df_for_pred.columns:
        print(f"Внимание: признак {feature} отсутствует в новых данных, будет создан с нулевыми значениями")
        df_for_pred[feature] = 0
missing_features = set(all_features) - set(df_for_pred.columns)
if missing_features:
    raise ValueError(f"Отсутствуют необходимые признаки: {missing_features}")
if best_model_name == 'LinearRegression':
    X_all = scaler.transform(df_for_pred[all_features])
else:
    X_all = df_for_pred[all_features]
original_df['ASAAK_pred'] = best_model.predict(X_all)
output_df = original_df.copy()

output_path = ('2onexo_analiz2.xlsx')
output_df.to_excel(output_path, index=False)

print("ПРОГРАММА ВЫПОЛЕНА")