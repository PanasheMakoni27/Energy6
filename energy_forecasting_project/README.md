# Energy Forecasting Project

This repository implements a 6-hour ahead energy forecasting pipeline using XGBoost. It includes data preprocessing, model training, evaluation, model persistence and a small Flask app for live forecasts.

## Repository Structure

- `energy_forecasting_project/`
  - `app.py` - Flask web application serving a dashboard and JSON forecast API.
  - `main.py` - Script that runs the full training pipeline (load, preprocess, train, evaluate, save models).
  - `notebooks/` - Jupyter notebooks for exploration and diagnostics (`exploration.ipynb`).
  - `models/` - Saved model artifacts (`xgb_model_t+1.pkl` ... `xgb_model_t+6.pkl`).
  - `requirements.txt` - Python dependencies used by the project.
  - `src/` - Source code for preprocessing and model training.
    - `preprocessing.py` - `create_features` and `create_targets` helpers.
    - `model_training.py` - `train_models`, `forecast_next_6_hours`, and `save_models` helpers.

## What it does

- Loads time-series energy data from a CSV (example path used in code: `C:/Users/prese/Downloads/Recruitment Dataset (1).csv`).
- Extracts time features: hour, day of week, and month.
- Creates lag features and rolling statistics.
- Generates multi-step targets `target_t+1` ... `target_t+6`.
- Trains six XGBoost regressors (one per forecast horizon) and evaluates with MAE.
- Saves trained models to `models/` and exposes a Flask dashboard for live predictions.

## Quickstart (local)

1. Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run training and evaluation (from project root):

```powershell
python energy_forecasting_project\main.py
```

3. Run the Flask app (for live forecasts):

```powershell
cd energy_forecasting_project
python app.py
# then open http://127.0.0.1:5000/
```

## Notes and TODOs

- Consider converting `create_features` into a reusable transformer and wrapping training in a `Pipeline` to ensure identical preprocessing for training and serving.
- Add time-series cross-validation (`TimeSeriesSplit`) and hyperparameter tuning (Optuna / RandomizedSearchCV) to improve model robustness.
- Add monitoring, model versioning (MLflow), and Dockerization for production deployment.

## License

This project does not include a license file. Add one if you intend to publish or share the code.
# Energy Forecasting Project

This project focuses on forecasting energy consumption using machine learning techniques. It includes data preprocessing, model training, and evaluation steps.

## Project Structure

- `data/` — Contains the energy dataset.
- `notebooks/` — Jupyter notebooks for exploration and analysis.
- `models/` — Trained machine learning models.
- `src/` — Source code for preprocessing and model training.
- `main.py` — Entry point for running the pipeline.
- `requirements.txt` — List of required Python packages.

## Getting Started

1. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
2. Run the main script:
   ```sh
   python main.py
   ```
