# Background Research — Wildfire Prediction with Machine Learning

## The Problem

Wildfires in the United States burn an average of 7 million acres annually (2010–2023), with costs exceeding $3 billion/year in suppression alone. Climate change is extending fire seasons and increasing severity. Current detection systems rely on:

- **Satellite imagery** (MODIS, VIIRS, GOES): Detect active fires but only after ignition; orbital timing creates detection delays of minutes to hours
- **Camera networks** (ALERTWildfire): Limited spatial coverage; dependent on visibility conditions
- **Lookout towers**: Declining in number; human-dependent
- **911 reports**: Reactive by nature; delay from ignition to report

**The gap**: No system effectively combines multiple pre-ignition signals to predict fires *before* they start. This project aims to fill that gap by fusing weather, fuel, topographic, atmospheric, and social signals into a predictive model.

---

## Key Fire Science Concepts

### Fire Triangle
Three conditions must be present for a fire:
1. **Fuel** — vegetation (type, moisture content, loading)
2. **Weather** — low humidity, high temperature, wind
3. **Ignition** — lightning, human activity

Our model must capture all three.

### Fire Weather Index (FWI) System
The Canadian FWI system is the international standard for fire danger rating. Components:

- **FFMC** (Fine Fuel Moisture Code): Moisture of litter and fine fuels (1-day lag)
- **DMC** (Duff Moisture Code): Moisture of moderate-depth organic layer (~15-day lag)
- **DC** (Drought Code): Deep soil/heavy fuel moisture (~52-day lag)
- **ISI** (Initial Spread Index): Fire spread rate (wind + FFMC)
- **BUI** (Build Up Index): Fuel available for burning (DMC + DC)
- **FWI** (Fire Weather Index): Overall fire danger intensity

gridMET provides pre-computed variants (Energy Release Component, Burning Index) for the US.

### Fuel Models
LANDFIRE classifies US vegetation into fuel models that predict fire behavior:
- **13 Anderson models**: Classic categories (grass, shrub, timber, slash)
- **40 Scott & Burgan models**: More detailed; used in modern fire modeling (FARSITE, FlamMap)

### Haines Index
Atmospheric stability and dryness index (scale 2–6). Higher values = more extreme fire behavior and plume-dominated events.

### Vapor Pressure Deficit (VPD)
The difference between how much moisture the air can hold and how much it holds. High VPD = aggressive drying of fuels. Research shows VPD is the single strongest weather predictor of fire activity in the western US.

---

## Prior Work in ML for Wildfire Prediction

### Key Papers and Approaches

**1. Jain et al. (2020) — "A review of machine learning applications in wildfire science and management"**
- Comprehensive review of ML approaches
- Found gradient boosting (XGBoost, LightGBM) consistently top-performing for fire occurrence prediction
- Emphasized importance of temporal features and class imbalance handling

**2. Radke et al. (2019) — "FireCast: Leveraging Deep Learning to Predict Wildfire Spread"**
- Used CNN on satellite imagery for fire spread prediction
- Achieved high accuracy but focused on spread, not ignition prediction

**3. Sayad et al. (2019) — "Predictive modeling of wildfires: A new dataset and ML approach"**
- Combined weather, topography, and vegetation features
- Random Forest achieved 95% AUC-ROC but with potential spatial leakage

**4. Coffield et al. (2019) — "Machine learning to predict final fire size"**
- Used weather, fuel, and topography to predict fire size at discovery
- XGBoost outperformed other methods; weather variables most important

### Common Findings Across Literature
1. **Gradient boosting dominates** tabular fire prediction tasks
2. **Weather variables** (especially VPD, humidity, wind) are consistently most important
3. **Temporal validation** is critical — random splits overestimate performance
4. **Class imbalance** requires careful handling; simple accuracy is misleading
5. **Spatial features** (topography, fuel type) improve generalization
6. **Human factors** are underrepresented in most models

---

## Fire Causes in the US (FPA-FOD Data)

Understanding ignition sources helps target features:

| Cause | % of Fires | % of Area Burned |
|-------|-----------|-----------------|
| Debris burning | 22% | 5% |
| Arson | 16% | 6% |
| Lightning | 15% | 56% |
| Equipment use | 10% | 3% |
| Campfire | 5% | 2% |
| Children | 5% | 1% |
| Smoking | 4% | 1% |
| Railroad | 3% | 1% |
| Powerline | 2% | 5% |
| Other/Unknown | 18% | 20% |

**Key insight**: Lightning causes only 15% of fires but 56% of area burned — these are large, remote fires. Human-caused fires are more frequent but generally smaller and near infrastructure.

---

## Geographic Focus Areas

Highest fire risk regions in the US (priority for model development):
1. **Pacific Northwest**: WA, OR — dry summers, beetle-killed timber
2. **California**: Year-round risk; Santa Ana winds; WUI interface
3. **Northern Rockies**: MT, ID — lightning-prone; remote
4. **Southwest**: AZ, NM — monsoon lightning; grassland fires
5. **Southeast**: FL, GA, SC — prescribed fire confusion; human-caused
6. **Great Plains**: TX, OK, KS — grassland fires; wind-driven

---

## Technical Considerations

### Spatial Resolution Trade-offs
- **Too fine (1km)**: Massive dataset; most cells always empty; harder to train
- **Too coarse (100km)**: Loses local variation; can't pinpoint risk
- **Sweet spot (~10km)**: Manageable size; captures local weather/fuel variation; matches gridMET resolution

### Temporal Resolution Trade-offs  
- **Hourly**: Captures diurnal weather cycles but massively increases data volume
- **Daily**: Good balance; matches most data source update frequencies
- **Weekly**: Loses day-to-day weather dynamics critical for fire weather

**Recommendation**: Daily temporal resolution, ~10km spatial grid, starting with contiguous US.

### Prediction Window
- **24-hour**: Highest accuracy; most actionable for fire managers
- **48-hour**: Moderate accuracy; useful for resource positioning
- **7-day**: Lower accuracy; useful for strategic planning
- **Seasonal**: Broad risk outlook; different modeling approach (climate-based)

**Approach**: Start with 7-day prediction window for initial model, then test shorter windows.
