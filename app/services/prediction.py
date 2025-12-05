import joblib
import numpy as np
import pandas as pd
from app.core.config import settings

# Load model ONCE
try:
    print(f"🔍 Attempting to load model from: {settings.BEST_MODEL_PATH}")
    model = joblib.load(settings.BEST_MODEL_PATH)
    print("✅ CatBoost model loaded successfully.")
except Exception as e:
    raise RuntimeError(f"❌ FAILED to load model: {e}\n"
                       f"Path: {settings.BEST_MODEL_PATH}\n"
                       f"Exists? {settings.BEST_MODEL_PATH.exists()}")

def predict_sales(preprocessed_df: pd.DataFrame) -> tuple[float, float]:
    """
    Returns log-scale and original-scale predictions.
    """
    try:
        log_pred = model.predict(preprocessed_df)[0]
        original_pred = np.exp(log_pred)
        return float(log_pred), float(original_pred)
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {e}")