# Seismic hazard forecasting for Turkey

This project builds a reproducible seismic hazard forecasting workflow for Turkey using AFAD earthquake data from 1990 to 2025. It estimates magnitude completeness, fits Gutenberg Richter relations, applies declustering, and runs a space time ETAS model to forecast future seismicity. Spatial forecasts use a strain weighted field so areas close to active faults have higher influence and more realistic patterns.

## Project goals

The aim is to understand broad spatial and temporal patterns in Turkish seismicity and create a transparent forecasting workflow that can be extended to future hazard studies. The model combines statistical methods with simple physical constraints so the forecast avoids random scatter and reflects real tectonic structure.

## Workflow

1. **Data preparation**  
   AFAD events from 1990 to 2025 are cleaned, filtered and transformed into a consistent format. Duplicate entries are removed, missing fields are handled and coordinates and magnitudes are standardised.

2. **Magnitude completeness estimation**  
   Magnitude of completeness is estimated using frequency magnitude curves. The linear section of the Gutenberg Richter distribution is identified so the model only uses magnitudes that are reliably recorded across Turkey.

3. **Gutenberg Richter fitting with AEReLU**  
   A Gutenberg Richter relation is fitted to the declustered mainshock catalogue. The b value is estimated using maximum likelihood. An Adaptive Exponential ReLU model implemented in PyTorch learns a smooth non linear approximation of the log frequency magnitude curve, which improves stability of the fit at high magnitudes. Diagnostic plots are generated to check fit quality.

4. **Declustering**  
   The Gardner Knopoff method is applied to remove aftershocks and isolate the independent mainshock catalogue that is used for completeness estimation and Gutenberg Richter fitting.

5. **Strain weighted spatial field**  
   A spatial grid is created across Turkey and each cell is assigned a strain weight based on distance to active faults. Cells closer to the North Anatolian and East Anatolian faults receive higher influence. This produces a realistic background field for forecasting.

6. **Space time ETAS simulation**  
   A strain weighted space time ETAS model simulates future seismicity for Turkey. Parameters control productivity, temporal decay and spatial spread. The strain field modifies the background rate so simulated events concentrate along fault systems instead of appearing uniformly.

7. **Visualisation**  
   Maps of simulated seismicity and spatial intensity are produced using GeoPandas and Matplotlib. The forecast patterns align with the North and East Anatolian faults and highlight regions with higher expected activity.


## Code structure

- `data/` AFAD data inputs  
- `notebooks/` full analysis and forecast workflow  
- `utils/` helper functions for completeness estimation, Gutenberg Richter fitting and mapping  
- `plots/` generated figures

## Requirements

The workflow uses Python, NumPy, Pandas, GeoPandas, PyTorch, Matplotlib and SciPy.
