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

### Dataset exploration & EDA (`exploration/eda.py`)

The EDA component processes every trip in the ``train`` and ``student_test``
splits and produces dataset-wide statistics, plots, interactive maps and a
report under ``outputs/eda/``.

```bash
python -m exploration.eda
```

It computes:

* Dataset inventory (trip counts, file presence, malformed records).
* GPS trajectory analysis (intervals, speed, gaps, duplicates, outliers).
* Road-network analysis (nodes/arcs, components, degree, self-loops, duplicates).
* Reference-route analysis (length, repeats, continuity, disconnections).
* GPS-to-road-network spatial analysis (KD-tree nearest-node distance).

Outputs include five CSV summaries, ten Matplotlib plots, Folium maps for
representative trips, and ``eda_report.md``.

### Candidate generation (`exploration/candidate_generation/`)

For every GPS observation, identifies nearby plausible road segments (arcs)
from the road network using a KD-tree spatial index and exact point-to-segment
distance calculations.

```bash
# Single trip
python -m exploration.candidate_generation.candidate_generator --split train --trip 00000001
python -m exploration.candidate_generation.candidate_generator --split student_test --trip 00000076

# All trips in a split (writes aggregate statistics CSV)
python -m exploration.candidate_generation.candidate_generator --split train --all-trips
python -m exploration.candidate_generation.candidate_generator --split student_test --all-trips
```

Options:

* ``--search-radius`` - search radius in metres (default: 50; try 25/50/75/100).
* ``--max-candidates`` - max candidates per observation (default: 5).
* ``--visualise`` - generate a Folium map.
* ``--all-trips`` - process every trip in the split.

It computes:

* Point-to-segment distance (not just node distance) with segment projection.
* GPS movement bearing and segment bearing.
* Distance along segment from start node.
* Per-trip miss rate, candidates per observation, and min/max candidates.

Candidate generation depends only on GPS observations and the road network.
Reference-route information is never used to generate candidates; it is used
solely for evaluation on training trips (reference-arc coverage and observation
reference coverage).

Outputs per trip: ``candidates.csv`` and ``candidates.jsonl`` under
``outputs/candidates/<split>/<trip_id>/``, plus optional Folium maps under
``outputs/candidates/maps/<split>/``. Aggregate statistics are written to
``outputs/candidates/candidate_statistics_<split>.csv``.

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
