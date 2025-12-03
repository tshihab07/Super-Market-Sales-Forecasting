import joblib
import pandas as pd
import numpy as np
from app.core.config import settings

# load model
try:
    model = joblib.load(settings.BEST_MODEL_PATH)
    print("✅ CatBoost model loaded successfully.")
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

def predict_sales(preprocessed_df: pd.DataFrame) -> tuple[float, float]:
    """
    Returns:
        log_prediction: float (log-scale)
        original_prediction: float (exp(log_prediction))
    """
    log_pred = model.predict(preprocessed_df)[0]
    original_pred = np.exp(log_pred)
    return float(log_pred), float(original_pred)