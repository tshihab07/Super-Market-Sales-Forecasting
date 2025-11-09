# utilities.py
# Reusable utilities for SuperMarket Sales Forecasting ML Models
# Designed for consistency, reproducibility, and extensibility

import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_validate
import warnings
warnings.filterwarnings("ignore")


# ------------------------------------------------------------------
# CLASS: Evaluator
# Unified metric calculation, train/test/CV comparison, and overfitting diagnosis
# ------------------------------------------------------------------
class Evaluator:
    
    @staticmethod
    def safe_mape(y_true, y_pred, epsilon=1e-8):
        """Robust MAPE: avoids division by zero."""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_true_safe = np.where(np.abs(y_true) < epsilon, epsilon, y_true)
        return np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100


    @staticmethod
    def calculate_metrics(y_true, y_pred):
        """Compute [MSE, MAE, RMSE, R2, MAPE] for regression."""
        if len(y_true) == 0 or len(y_pred) == 0:
            raise ValueError("Empty arrays passed to calculate_metrics.")
        
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        mape = Evaluator.safe_mape(y_true, y_pred)
        
        return [mse, mae, rmse, r2, mape]
    

    @staticmethod
    def create_performance_table(train_metrics, test_metrics):
        """Return DataFrame: Metrics | Training | Test """
        perf_df = pd.DataFrame({
            'Metrics': ['MSE', 'MAE', 'RMSE', 'R2 Score', 'MAPE'],
            'Training': train_metrics,
            'Test': test_metrics
        }).round(4)
        return perf_df


    @staticmethod
    def cv_evaluate(model, X, y, cv, scoring=None):
        """Run cross-validation and return dict of average CV metrics (MSE, MAE, RMSE, R2, MAPE)."""
        if scoring is None:
            scoring = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']
        
        # Standard metrics via cross_validate
        cv_results = cross_validate(
            model, X, y, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False
        )
        
        cv_mse = -cv_results['test_neg_mean_squared_error'].mean()
        cv_mae = -cv_results['test_neg_mean_absolute_error'].mean()
        cv_rmse = np.sqrt(cv_mse)
        cv_r2 = cv_results['test_r2'].mean()
        
        # MAPE: custom loop
        mape_scores = []
        for train_idx, val_idx in cv.split(X, y):
            model_clone = model
            # For safety, fit clone
            try:
                model_clone.fit(X.iloc[train_idx], y.iloc[train_idx])
                y_pred = model_clone.predict(X.iloc[val_idx])
            
            except Exception:
                # Fallback: use original model fit if stateful
                y_pred = model.predict(X.iloc[val_idx])
           
            mape_scores.append(Evaluator.safe_mape(y.iloc[val_idx], y_pred))
        
        cv_mape = np.mean(mape_scores)
        
        return {
            'CV MSE': cv_mse,
            'CV MAE': cv_mae,
            'CV RMSE': cv_rmse,
            'CV R2': cv_r2,
            'CV MAPE': cv_mape
        }
    
    @staticmethod
    def assess_overfitting(cv_r2, test_r2, cv_rmse, test_rmse, tolerance=0.05):
        """Determine overfitting status and generalization quality."""
        r2_gap = cv_r2 - test_r2
        rmse_ratio = test_rmse / cv_rmse if cv_rmse > 0 else np.inf
        
        # Overfitting logic
        if r2_gap > tolerance or rmse_ratio > 1.05:
            overfit_status = "Yes"
        
        elif abs(r2_gap) <= tolerance and 0.95 <= rmse_ratio <= 1.05:
            overfit_status = "No"
        
        else:
            overfit_status = "Mild"
        
        # Generalization status
        if test_r2 > 0.85:
            gen_status = "Excellent"
        
        elif test_r2 > 0.70:
            gen_status = "Good"
        
        elif test_r2 > 0.50:
            gen_status = "Fair"
        
        else:
            gen_status = "Poor"
        
        return overfit_status, gen_status


# ------------------------------------------------------------------
# CLASS: ModelPersister
# Handles saving models, performance summaries, and aggregated comparisons
# ------------------------------------------------------------------
class ModelPersister:
    
    def __init__(self, model_name, artifacts_root="../artifacts"):
        self.model_name = model_name
        self.root = Path(artifacts_root)
        self.model_dir = self.root / "models"
        self.perf_dir = self.root / "model-performance"
        
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.perf_dir.mkdir(parents=True, exist_ok=True)
    
    
    def save_model(self, model, model_type="sklearn"):
        """Save model. Supports sklearn (joblib) and Keras (if needed later)."""
        if model_type.lower() in ["keras", "tf", "tensorflow"]:
            model_path = self.model_dir / f"{self.model_name.lower()}_model.keras"
            model.save(model_path)
        
        else:
            model_path = self.model_dir / f"{self.model_name.lower()}_model.pkl"
            joblib.dump(model, model_path)
        print(f"✅ Model saved: {model_path}")
    
    
    def save_performance_csv(self, perf_df, suffix="performance"):
        """Save individual model performance (train/test)."""
        path = self.perf_dir / f"{self.model_name.lower()}_{suffix}.csv"
        perf_df.to_csv(path, index=False)
        print(f"✅ Performance saved: {path}")
    
    
    def append_to_aggregated(self, df, filename="a_ModelPerformance.csv"):
        """Append this model's row to shared performance CSV (for multi-model comparison)."""
        path = self.perf_dir / filename
        if path.exists():
            existing = pd.read_csv(path)
            df = pd.concat([existing, df], ignore_index=True)
        df.to_csv(path, index=False)
        print(f"✅ Appended to aggregated file: {path}")
    
    
    def append_overfitting_row(self, df):
        """Append overfitting analysis row (extensible for future models)."""
        self.append_to_aggregated(df, filename="a_OverfittingAnalysis.csv")


# ------------------------------------------------------------------
# CLASS: DataHandler
# For loading/splitting — mirrors your structured workflow
# ------------------------------------------------------------------
class DataHandler:
    
    @staticmethod
    def load_dataset(file_path):
        """Load CSV and return df, X, y (with 'OutletSales' as target)."""
        df = pd.read_csv(file_path)
        X = df.drop(columns=['OutletSales'], errors='ignore')
        y = df['OutletSales']
        return df, X, y
    

    @staticmethod
    def load_artifacts(artifacts_dir, cv_required=True):
        """Load x_train, x_test, y_train, y_test, [cv]."""
        artifacts_dir = Path(artifacts_dir)
        artifacts = {
            'x_train': joblib.load(artifacts_dir / "x_train.pkl"),
            'x_test': joblib.load(artifacts_dir / "x_test.pkl"),
            'y_train': joblib.load(artifacts_dir / "y_train.pkl"),
            'y_test': joblib.load(artifacts_dir / "y_test.pkl")
        }

        if cv_required:
            try:
                artifacts['cv'] = joblib.load(artifacts_dir / "cv.pkl")
            
            except FileNotFoundError:
                print("⚠️ Warning: cv.pkl not found. Using default 5-Fold CV.")
                from sklearn.model_selection import KFold
                artifacts['cv'] = KFold(n_splits=5, shuffle=True, random_state=42)
        
        return artifacts