
"""Model training helpers for the energy forecasting project.

This module provides utilities to train one XGBoost regressor per
forecast horizon (t+1 ... t+6), evaluate with MAE, forecast using
trained models, and persist models to disk.

The changes here are purely documentation / comment cleanups.
"""

import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import joblib
import os


def train_models(X_train, y_train, X_test, y_test):
    """Train one XGBoost model per forecast step.

    Args:
        X_train: DataFrame of training features
        y_train: DataFrame of training targets (each column is a horizon)
        X_test: DataFrame of test features
        y_test: DataFrame of test targets

    Returns:
        List of trained XGBoost models (one per horizon)
    """
    models = []
    n_steps = y_train.shape[1]
    for i in range(n_steps):
        model = xgb.XGBRegressor()
        model.fit(X_train, y_train.iloc[:, i])

        # Evaluate on the test set and print MAE for this horizon
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test.iloc[:, i], preds)
        print(f"Step {i+1} MAE: {mae:.2f}")

        models.append(model)
    return models


def forecast_next_6_hours(latest_row, models):
    """Generate 6-step forecasts using a list of trained models.

    Args:
        latest_row: Series or 1D array of feature values for the latest timestep
        models: list of trained models in order (t+1 ... t+6)

    Returns:
        List of predictions (floats) for each horizon
    """
    predictions = []
    for model in models:
        pred = model.predict(latest_row.values.reshape(1, -1))
        predictions.append(pred[0])
    return predictions


def save_models(models, path_prefix="models/xgb_model_t+"):
    """Save each model to disk using joblib.

    The `path_prefix` should include the directory and the filename prefix
    (e.g. "models/xgb_model_t+"). Files will be written as
    "{path_prefix}1.pkl", "{path_prefix}2.pkl", ...
    """
    save_dir = os.path.dirname(path_prefix)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for i, model in enumerate(models):
        joblib.dump(model, f"{path_prefix}{i+1}.pkl")
