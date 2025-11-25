### Model Comparison Summary Table
Best Model by Comparison

| **Comparison Category**                             | **Winner Model**                               |
| --------------------------------------------------- | ---------------------------------------------- |
| Test R² (↑ better)                                  | CatBoost                                       |
| CV R² (↑ better)                                    | CatBoost                                       |
| Test MAPE (%) (↓ better)                            | CatBoost, LightGBM, GBR                        |
| CV RMSE (↓ better)                                  | CatBoost                                       |
| Predicted vs Actual (High Value)                    | CatBoost *(tightest clustering near diagonal)* |
| CV R² vs Test R² (Overfitting Indicator)            | RandomForest *(smallest gap)*                  |
| Cross-Validation Robustness (Mean ± Std)            | CatBoost *(highest mean, lowest std)*          |
| Overfitting Analysis (Generalization Indicator)     | LightGBM *(low overfitting, smallest R² gap)*  |
| Test MAPE (Balanced Performance Metric)             | CatBoost, LightGBM, GBR *(all 5.89%)*          |
| Composite Score Ranking (Final)                     | **CatBoost (0.596)**                           |


**Final Recommendation: Select CatBoost**

Based on a comprehensive evaluation across accuracy, robustness, generalization, and business impact, CatBoost emerges as the optimal model for deployment in your Supermarket Sales Forecasting system.s