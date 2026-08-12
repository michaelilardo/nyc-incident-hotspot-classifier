# Can Small Interpretable Models Accurately Identify Crime Hotspot Patterns?

Team project (DIMACS Practicum) — Vazken Hajinelian, Tyler Rice, Michael Ilardo

## Problem
Cities rely on predictive models to allocate police resources, but these decisions carry real consequences — misallocated patrols and eroded public trust. Departments face a tradeoff: interpretable models are auditable but may miss complex patterns; black-box models perform better but can't explain why an area was flagged. This project tests whether small, interpretable models can match black-box accuracy on real NYC crime data, and quantifies what transparency actually costs.

## Data
Three NYC Open Data sources (2022–2024), aggregated to precinct-week/month observations:
- **NYPD Complaints**: 3,100+ incidents, 16.4% hotspot rate
- **311 Service Requests**: 9.8M complaints, 25.8% hotspot rate, 12,570 precinct-week observations
- **Shooting Incidents**: 3,167 shootings, 9.25% hotspot rate, 12,000 precinct-week observations

## Approach
- Built 19 models total across three interpretability tiers: interpretable (Decision Tree, Threshold Guess Binarizer), semi-interpretable (Random Forest, Gradient Boosting, XGBoost, Logistic Regression), and black-box (Neural Network, Linear SVM, SVM-RBF)
- Engineered time-based features (4wk/13wk rolling averages, trends) with strict past-vs-future separation
- Temporal train-test split (train 2022–2023, test 2024) to prevent seasonal leakage
- **Caught and corrected a data leakage bug**: initial rule-derived features let models score ~1.00 accuracy by learning the hotspot label definition itself rather than real patterns. Removed those features, rebuilt on true time-based aggregates — realistic accuracy dropped to ~0.81 (Gradient Boosting best), confirming the models now learn genuine signal
- Evaluated using PR-AUC (not just ROC-AUC/accuracy) since datasets were class-imbalanced — PR-AUC isolates minority-class (hotspot) detection specifically
- Spatial validation: 250m grid cells, mapped predicted risk against actual 2024 incidents citywide

## Results
| Dataset | Best Model | Accuracy | Winner |
|---|---|---|---|
| NYPD Complaints | Decision Tree | 76.5% | Interpretable — black-box models actually underperformed |
| 311 Requests | Neural Net / SVM | 97.7% | Black-box — 30-point gap over interpretable |
| Shootings | XGBoost | 88.5% acc / low F1 | Neither — all models struggle on rare-event detection |

Key finding: **no single model type wins everywhere.** The right choice depends on the dataset's structure and what's at stake — for high-accountability decisions (complaints, patrol allocation), a simple 2-feature decision tree matched or beat every complex model. For low-stakes volume decisions (311), black-box models were clearly worth the opacity. For the highest-stakes case (shootings), no model was good enough yet — the bottleneck was data richness, not algorithm choice.

## Tools
Python, scikit-learn, XGBoost, pandas, SHAP, geopandas

## Team Contribution
I led the shooting incidents dataset — built the decision tree and feature pipeline (8-week rolling shooting mean as the dominant signal), trained/evaluated 6 models, and ran the spatial validation (250m grid, decile capture analysis showing top 10% of cells capture 45% of actual shootings).
