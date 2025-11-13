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
    def performance_table(train_metrics, test_metrics):
        """Return DataFrame: Metrics | Training | Test """
        perf_df = pd.DataFrame({
            'Metrics': ['MSE', 'MAE', 'RMSE', 'R2 Score', 'MAPE'],
            'Training': train_metrics,
            'Test': test_metrics
        }).round(4)

        return perf_df
    

    @staticmethod
    def summary_builder(model_names, cv_df, test_metrics):
        """ Overall Model Performance (CV + Test) — Merged """
        test_df = pd.DataFrame({
            "Model": model_names,
            "Test MSE": [m[0] for m in test_metrics],
            "Test MAE": [m[1] for m in test_metrics],
            "Test RMSE": [m[2] for m in test_metrics],
            "Test R2": [m[3] for m in test_metrics],
            "Test MAPE": [m[4] for m in test_metrics]
        })

        merged = pd.merge(cv_df, test_df, on="Model", how="inner")
        return merged[[
            "Model",
            "CV MSE", "CV MAE", "CV RMSE", "CV R2", "CV MAPE",
            "Test MSE", "Test MAE", "Test RMSE", "Test R2", "Test MAPE"
        ]].round(4)


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
            overfit_status = "High"
        
        elif abs(r2_gap) <= tolerance and 0.95 <= rmse_ratio <= 1.05:
            overfit_status = "Low"
        
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
        
        return r2_gap, rmse_ratio, overfit_status, gen_status


# ------------------------------------------------------------------
# CLASS: ModelPersister
# Handles saving models, performance summaries, and aggregated comparisons
# ------------------------------------------------------------------
# Handles saving trained models and performance results to organized directories
class ModelPersister:
    
    def __init__(self, model_name, artifacts_root="../artifacts"):
        self.model_name = model_name
        self.artifacts_root = Path(artifacts_root)
        self.model_dir = self.artifacts_root / "models"
        self.performance_dir = self.artifacts_root / "model-performance"
        
        # Create directories
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.performance_dir.mkdir(parents=True, exist_ok=True)
    

    # Save the trained model in appropriate format
    def save_model(self, model):
        joblib.dump(model, self.model_dir / f"model_{self.model_name.title()}.pkl")
        
        print(f"Model saved: {self.model_dir}/{self.model_name.lower()}.pkl")

    # Save full train/test/CV metrics for this model only
    def save_performance(self, performance_df, tag=""):
        if tag:
            filename = f"{self.model_name.lower()}{tag}.csv"
        else:
            filename = f"{self.model_name.lower()}Performance.csv"
        
        path = self.performance_dir / filename
        performance_df.to_csv(path, index=False)
        print(f"{self.model_name} performance saved: {path}")
    

    # Append this model's summary metrics to the shared performance file
    def aggregated_performance(self, df):
        path = self.performance_dir / "a_ModelPerformance.csv"
        
        # Append or create
        if path.exists():
            model_perf = pd.read_csv(path)                          # open previous loaded data
            df = pd.concat([model_perf, df], ignore_index=True)     # append new data
            df.to_csv(path, index=False)
        
        else:
            df.to_csv(path, index=False)
        
        print(f"Appended to aggregated performance: {path}")
    

    # Append this model's overfitting metrics to the shared overfitting file
    def append_overfitting(self, df):
        path = self.performance_dir / "a_overfittingAnalysis.csv"
        
        if path.exists():
            overfit_df = pd.read_csv(path)                          # open previous loaded data
            df = pd.concat([overfit_df, df], ignore_index=True)     # append new data
            df.to_csv(path, index=False)
        
        else:
            df.to_csv(path, index=False)
        
        print(f"Appended to overfitting analysis: {path}")


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