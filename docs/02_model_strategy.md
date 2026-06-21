# Model Strategy — Wildfire Risk Prediction

## Problem Formulation

**Task**: Binary classification — given a spatial grid cell and time window, predict whether a wildfire will ignite.

**Target Variable**: Fire occurrence (1/0) derived from FPA-FOD historical records and NASA FIRMS active fire detections.

**Spatial Unit**: Grid cells (e.g., 10km × 10km) covering the contiguous US.

**Temporal Unit**: Daily or weekly prediction windows.

**Key Challenge**: Severe class imbalance — fires are rare events (~0.1% of grid-cell-days).

---

## Feature Categories

### Dynamic Features (time-varying)
- Temperature (max, min, mean)
- Relative humidity (min is most critical)
- Wind speed and direction
- Precipitation (accumulated over 1, 3, 7, 14 days)
- Fire Weather Index (FWI) components: FFMC, DMC, DC, ISI, BUI, FWI
- Lightning strike count and density (1, 3, 7 day windows)
- Soil moisture (SMAP)
- NDVI (vegetation greenness/stress)
- Air quality (PM2.5, AQI) — lagged and rate of change
- Drought severity index
- Day of year / season (cyclical encoding)

### Static Features (location-dependent)
- Elevation, slope, aspect
- Fuel model type (LANDFIRE 13/40 models)
- Canopy cover and height
- Land cover type
- Distance to roads, urban areas, campgrounds
- Historical fire frequency for the grid cell

### Derived Features
- Consecutive dry days
- Vapor pressure deficit (VPD)
- Haines Index (atmospheric stability + dryness)
- Rate of change in humidity, temperature
- Rolling fire weather severity indices
- Spatial lag features (conditions in neighboring cells)

---

## Candidate Models

### 1. Random Forest (Baseline)
- **Strengths**: Handles mixed feature types, robust to outliers, built-in feature importance, minimal tuning
- **Weaknesses**: Can underperform on tabular data vs. boosting, limited temporal pattern capture
- **Use**: Establish baseline performance; identify important features early

### 2. XGBoost
- **Strengths**: State-of-the-art for tabular data, handles imbalanced classes (scale_pos_weight), fast training, built-in regularization
- **Weaknesses**: Requires more tuning than RF
- **Use**: Primary candidate for production model

### 3. LightGBM
- **Strengths**: Faster than XGBoost on large datasets, native categorical feature support, lower memory usage
- **Weaknesses**: Can overfit with small data; leaf-wise growth more sensitive to hyperparameters
- **Use**: Compare against XGBoost; may be preferred for larger datasets

### 4. LSTM (Long Short-Term Memory)
- **Strengths**: Captures temporal sequences (e.g., drying trends over days/weeks), learns complex non-linear temporal dependencies
- **Weaknesses**: Requires more data, harder to train, less interpretable, doesn't natively handle spatial features well
- **Use**: Assess whether temporal patterns improve prediction over snapshot-based models

### 5. Logistic Regression (Interpretable Baseline)
- **Strengths**: Fully interpretable, fast, establishes a floor for comparison
- **Weaknesses**: Assumes linear relationships; won't capture interactions without manual feature engineering
- **Use**: Sanity check; if this performs well, the problem may be simpler than expected

---

## Evaluation Framework

### Metrics
| Metric | Purpose |
|--------|---------|
| AUC-ROC | Primary: discrimination ability across thresholds |
| AUC-PR | Better than ROC for imbalanced data |
| Precision @ threshold | False alarm rate at operating point |
| Recall @ threshold | Detection rate at operating point |
| F1 Score | Balance of precision and recall |
| Brier Score | Calibration of probability estimates |

### Validation Strategy
- **Temporal split**: Train on years 1992–2016, validate on 2017–2018, test on 2019–2020
- **Never** random split — temporal leakage would inflate metrics
- **Spatial cross-validation**: Block cross-validation by geographic region to test generalization

### Class Imbalance Handling
1. **Cost-sensitive learning**: Adjust class weights (XGBoost: `scale_pos_weight`)
2. **SMOTE / undersampling**: For tree models; compare oversampled vs weighted approaches
3. **Focal loss**: For neural network models
4. **Threshold tuning**: Optimize classification threshold on validation set for desired precision/recall trade-off

---

## Visualization Plan

### Exploratory Data Analysis
- **Correlation heatmaps**: All features against each other and against fire occurrence
- **Distribution plots**: Feature distributions for fire vs. no-fire samples
- **Time series**: Seasonal patterns in fire occurrence and weather variables
- **Geographic maps**: Fire density by region; feature spatial distributions

### Model Analysis
- **Feature importance**: SHAP values for each model
- **Partial dependence plots**: How individual features affect prediction
- **Calibration curves**: Are predicted probabilities reliable?
- **Confusion matrices**: At selected operating thresholds
- **ROC / PR curves**: Compare all candidate models on same plot
- **Error analysis**: Where/when does the model fail? Geographic/temporal patterns in errors

---

## Development Phases

### Phase 1: Data Collection & EDA (Current)
- Acquire priority data sources
- Build data ingestion pipeline
- Exploratory analysis and visualization
- Feature engineering

### Phase 2: Baseline Models
- Train Random Forest and Logistic Regression baselines
- Establish baseline metrics
- Identify most important features

### Phase 3: Advanced Models
- Train XGBoost, LightGBM, LSTM
- Hyperparameter tuning (Optuna or grid search)
- Model comparison across all metrics

### Phase 4: Model Selection & Refinement
- Select best model(s)
- Ensemble if beneficial
- Calibrate probability outputs
- Error analysis and feature iteration

### Phase 5: Dashboard Integration
- Wrap selected model in prediction API
- Build geographic dashboard with live data feeds
- Implement risk zone visualization (red/yellow/green)
- Configure forward-looking prediction windows
