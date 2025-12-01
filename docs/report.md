# Conclusion: Supermarket Sales Forecasting

---

### Best Model: **CatBoost (BayesSearchCV)**

---

#### **Summary of the Modeling Process**

This project implemented and evaluated five state-of-the-art machine learning models —

- **XGBoost**
    - Baseline
    - Optuna
    - RandomSearchCV
- **Random Forest**
    - Baseline
    - Optuna
    - HalvingRandomSearchCV
- **LightGBM**
    - Baseline
    - Optuna
    - RandomSearchCV
- **CatBoost**
    - Baseline
    - Optuna
    - BayesianSearch
- **Gradient Boosting Regressor**
    - Baseline
    - Optuna
    - RandomizedSearchCV

to predict `OutletSales` using historical supermarket transaction data. Each model family underwent rigorous hyperparameter optimization using appropriate techniques (`Optuna`, `RandomizedSearchCV`, `HalvingRandomSearchCV`, and `BayesianSearchCV`) with `5-fold` cross-validation. Comprehensive regression metrics — including `Test R²`, `CV R²`, `MAPE`, `RMSE`, and `overfitting gap` (R² Gap) were calculated for all variants. Visual diagnostics, including `predicted vs. actual` plots, `residual` analysis, and `Q-Q plots`, ensured model validity and business readiness.

---

#### **Overall Model Review**
- Among all 15 variants, ***CatBoost (BayesianSearch)*** achieved the highest `Test R² (0.7466)` and lowest `Test MAPE (5.886%)`, indicating superior predictive accuracy and business-relevant precision for inventory planning.
- It exhibits excellent calibration and generalization, with a minimal `overfitting gap (R² Gap = −0.0121)` and strong cross-validation stability (`CV R² = 0.7345`, `CV RMSE = 0.5222`).
- ***LightGBM (RandomSearchCV)*** and ***GBR (RandomizedSearchCV)*** tie for second place (`Test R² = 0.7458`, `MAPE = 5.890%`), with even lower `overfitting gaps (−0.0127, −0.0124)` it is ideal for high-stakes forecasting where stability is paramount.
- ***XGBoost (RandomSearchCV)*** ranks fourth in accuracy (`Test R² = 0.7455`) but achieves the smallest `overfitting gap (−0.0113)` and highest `RMSE` consistency (`Ratio = 0.992`), making it the most reliable for unseen store configurations.
- ***RandomForest (HalvingRandomSearchCV)*** shows the lowest `overfitting gap (−0.0100)` near-perfect generalization, though at a slight cost in absolute accuracy (`Test R² = 0.7410`).
- All top 5 models demonstrate `Low` overfitting and `Good` generalization, confirming robust model selection and tuning.
- Optuna-tuned variants consistently underperformed their Bayesian/Random counterparts suggesting over-regularization or premature pruning in aggressive search settings.

Therefore, **CatBoost (BayesianSearch)** is selected as the final, production-ready model for intelligent sales forecasting.

---

#### **Best Performing Model Performance Summary**

Based on a Test R² score, Test MAPE and overfitting gap, **CatBoost (BayesianSearchCV)** emerged as the superior model:

- `CV MSE`: 0.2727
- `CV MAE`: 0.4029
- `CV RMSE`: 0.5222
- `CV R2`: 0.7345
- `CV MAPE`: 5.847
- `Test MSE`: 0.2675
- `Test MAE`: 0.4
- `Test RMSE`: 0.5172
- `Test R2`: 0.7466
- `Test MAPE`: 5.8857
- `R2 Gap`: -0.0121
- `RMSE Ratio`: 0.9904
- `Overfitting Status`: Low
- `Model Status (Generalization)`: Good
- `Composite Score`: 0.59607

---

#### **Key Observations**

Accuracy-Stability Trade-off: While `CatBoost` leads in raw accuracy, `XGBoost` and `LightGBM` offer marginally better stability — valuable for rolling forecasts across diverse outlet types.
Categorical Advantage: `CatBoost`’s native handling of categorical features (e.g., `Item_Type`, `Outlet_Location`) likely contributed to its edge — reducing encoding bias.
No Severe Overfitting: All top models show `R² Gap < 0.03`, confirming effective regularization and validation strategy.
Optuna Caution: `Optuna`’s aggressive pruning may oversimplify complex sales patterns; Bayesian/Random search provided better exploration-exploitation balance.
Interpretability: `SHAP` analysis confirms intuitive drivers: `MRP`, `ItemWeight`, and `Outlet_Type` dominate predictions aligning with domain expertise.

---

#### **Final Recommendation**

Given the critical need for accurate, stable, and explainable sales forecasts in retail operations, `CatBoost (BayesianSearch)` is recommended for production deployment.

The model has been:

- ✅ Saved as `model_Catboost.pkl`
- ✅ Validated via residuals, Q-Q plots, and SHAP
- ✅ Optimized for MAPE (business-aligned error)
- ✅ Preprocessed consistently (categorical handling preserved)

---

#### Future enhancements:

- Store-level forecasting with hierarchical modeling
- Time-aware features (e.g., seasonality, holidays) for next-phase time-series extension
- Uncertainty quantification via quantile CatBoost (e.g., 90% prediction intervals)
- Automated retraining pipeline with drift detection (e.g., Evidently AI)
- Dashboard integration (Plotly/Dash) for real-time forecast monitoring
- This system empowers supermarket managers to move from reactive to predictive decision-making optimizing stock, reducing waste, and maximizing revenue.