from flask import Flask, render_template, jsonify
import pandas as pd
import joblib
import os
from src.preprocessing import create_features
from src.model_training import forecast_next_6_hours

app = Flask(__name__)

# Cache for loaded models
model_cache = None

# Error handling for loading latest data
def load_latest_data():
    csv_path = r"C:/Users/prese/Downloads/Recruitment Dataset (1).csv"
    try:
        df = pd.read_csv(csv_path, parse_dates=['date'])
        df = create_features(df)
        latest_row = df.drop(columns=['date', 'power_x']).iloc[-1]
        return latest_row
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def load_models():
    global model_cache
    if model_cache is not None:
        return model_cache
    models = []
    try:
        for i in range(1, 7):
            model = joblib.load(f"models/xgb_model_t+{i}.pkl")
            models.append(model)
        model_cache = models
        return models
    except Exception as e:
        print(f"Error loading models: {e}")
        return None

@app.route('/')
def index():
    latest_row = load_latest_data()
    models = load_models()
    if latest_row is None or models is None:
        return render_template('error.html', message="Could not load data or models. Please check your setup."), 500
    predictions = forecast_next_6_hours(latest_row, models)
    hours = [f"t+{i}" for i in range(1, 7)]
    return render_template('dashboard.html', predictions=predictions, hours=hours)

@app.route('/api/forecast')
def api_forecast():
    latest_row = load_latest_data()
    models = load_models()
    if latest_row is None or models is None:
        return jsonify({"error": "Could not load data or models."}), 500
    predictions = forecast_next_6_hours(latest_row, models)
    return jsonify({"predictions": predictions})

@app.errorhandler(404)
def not_found(e):
    return render_template('error.html', message="Page not found."), 404

@app.errorhandler(500)
def internal_error(e):
    return render_template('error.html', message="An internal error occurred."), 500

if __name__ == '__main__':
    app.run(debug=True)
