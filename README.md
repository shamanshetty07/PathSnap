# PathSnap

## Project Overview

PathSnap is a GPS map-matching project that identifies the most likely road segments travelled by a vehicle using GPS trajectory data and a road network.

## Project Pipeline

1. GPS dataset loading and exploratory data analysis
2. GPS data cleaning and preprocessing
3. Road-network preparation
4. Candidate road-segment generation
5. Feature extraction
6. Machine learning model training and comparison
7. Hyperparameter optimisation using Optuna
8. HMM-based map matching using Viterbi decoding
9. Evaluation and visualisation
10. FastAPI backend and Streamlit frontend

## Reference Route Reconstruction & Visualisation (`exploration/`)

The first implemented component loads a trip, reconstructs its reference route
from the ordered arc IDs in `.route`, validates route continuity, and produces
an interactive Folium map.

### CLI usage

```bash
python -m exploration.visualize_route --split train --trip 00000001
python -m exploration.visualize_route --split student_test --trip 00000076
```

Each run:

1. Loads `.track`, `.nodes`, `.arcs` and `.route` for the selected trip.
2. Reconstructs the reference route in arc order (repeated arc IDs preserved).
3. Validates continuity and reports disconnected transitions.
4. Saves an HTML map to `outputs/maps/<split>_<trip>_reference_route.html`.
5. Prints a trip analysis summary.

### Module layout

- `exploration/data_loader.py` — loads and validates the four trip files.
- `exploration/route_reconstruction.py` — builds the geographical route.
- `exploration/route_validation.py` — continuity checks and disconnection detection.
- `exploration/visualize_route.py` — Folium visualisation and CLI entry point.

## Technology Stack

- Python
- Pandas and NumPy
- GeoPandas, Shapely and PyProj
- Scikit-learn, XGBoost, LightGBM and CatBoost
- Optuna
- Hidden Markov Models
- Folium
- FastAPI
- Streamlit

## Project Structure

- data/: Raw and processed datasets
- notebooks/: Exploratory notebooks
- src/: Main project source code
- api/: FastAPI backend
- ui/: Streamlit frontend
- scripts/: Utility scripts
- results/: Evaluation results
- maps/: Generated maps
- artifacts/: Saved model artifacts
- reports/: Project documentation
- tests/: Automated tests

## Setup

Create and activate a Python virtual environment, then install dependencies:

1. py -3.11 -m venv venv
2. .\venv\Scripts\Activate.ps1
3. python -m pip install -r requirements.txt
