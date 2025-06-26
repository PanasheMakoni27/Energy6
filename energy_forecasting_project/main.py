import pandas as pd
from src.preprocessing import create_features, create_targets
from src.model_training import train_models, forecast_next_6_hours, save_models
from sklearn.model_selection import train_test_split

# Load data
csv_path = r"C:/Users/prese/Downloads/Recruitment Dataset (1).csv"
df = pd.read_csv(csv_path, parse_dates=['date'])
print('After loading:', df.shape)
print('NaN in power_x:', df['power_x'].isna().sum())
print('Non-numeric in power_x:', (~pd.to_numeric(df['power_x'], errors='coerce').notna()).sum())

# Feature engineering
df = create_features(df)
print('After create_features:', df.shape)
df = create_targets(df, n_steps=6)
print('After create_targets:', df.shape)

# Prepare features and targets
features = df.drop(columns=['date', 'power_x'] + [f'target_t+{i}' for i in range(1, 7)])
print('Features shape:', features.shape)
targets = df[[f'target_t+{i}' for i in range(1, 7)]]
print('Targets shape:', targets.shape)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, shuffle=False)
print('Train shape:', X_train.shape, y_train.shape)
print('Test shape:', X_test.shape, y_test.shape)

# Train models
models = train_models(X_train, y_train, X_test, y_test)

# Save models
save_models(models)

# Forecast and evaluate for the first test sample
true_vals = y_test.iloc[0]
pred_vals = forecast_next_6_hours(X_test.iloc[0], models)

import matplotlib.pyplot as plt
plt.plot(range(1, 7), true_vals, label='Actual')
plt.plot(range(1, 7), pred_vals, label='Predicted')
plt.xlabel("Hours Ahead")
plt.ylabel("Energy (MW)")
plt.legend()
plt.title("6-Hour Ahead Forecast")
plt.show()
