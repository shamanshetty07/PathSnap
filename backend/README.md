# PathSnap: Distributed Map-Matching and Route Reconstruction

## Overview

PathSnap reconstructs realistic vehicle trajectories from noisy GPS observations using a complete ML pipeline including candidate generation, feature engineering, Random Forest emission probabilities, and HMM/Viterbi decoding.

## Project Structure

```
PathSnap/
├── backend/                    # FastAPI backend
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py            # Main FastAPI application
│   │   ├── config.py          # Environment configuration
│   │   ├── api/               # API routes
│   │   │   ├── health.py      # Health check endpoint
│   │   │   ├── trips.py       # Trip management endpoints
│   │   │   ├── map_matching.py # Map matching endpoint
│   │   │   ├── results.py     # Route results endpoint
│   │   │   └── evaluation.py  # Evaluation metrics endpoint
│   │   ├── schemas/           # Pydantic schemas
│   │   │   ├── __init__.py
│   │   │   ├── trip.py        # Trip schemas
│   │   │   ├── map_matching.py # Map matching schemas
│   │   │   ├── route.py       # Route and geometry schemas
│   │   │   └── evaluation.py  # Evaluation schemas
│   │   ├── services/          # Business logic services
│   │   │   ├── __init__.py
│   │   │   ├── trip_service.py
│   │   │   ├── map_matching_service.py
│   │   │   └── route_service.py
│   │   ├── utils/             # Utility modules
│   │   │   ├── __init__.py
│   │   │   └── validators.py
│   │   └── models/            # ML model integration
│   │       └── hmm.py         # HMM map matcher
│   ├── requirements.txt       # Python dependencies
│   └── .env.example           # Environment variables
├── Data/                      # Dataset directory
│   ├── train/                 # Training trips
│   │   └── <trip_id>/
│   │       ├── .track
│   │       ├── .nodes
│   │       ├── .arcs
│   │       └── .route
│   └── student_test/          # Student test trips
│       └── <trip_id>/
│           ├── .track
│           ├── .nodes
│           ├── .arcs
│           └── .route
├── outputs/                   # Generated outputs
│   ├── models/                # Trained models
│   ├── map_matching/          # Map matching results
│   └── features/              # Engineered features
└── exploration/               # ML pipeline modules
```

## Installation

1. Clone the repository
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r backend/requirements.txt
   ```
4. Copy the environment example and configure:
   ```bash
   cp backend/.env.example backend/.env
   # Edit .env as needed
   ```
5. Ensure the trained model exists:
   ```bash
   # The model should be at outputs/models/random_forest.joblib
   # Train it if needed using: python -m models.train
   ```

## Running the Backend

Start the FastAPI server:

```bash
source venv/bin/activate
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The backend will be available at `http://0.0.0.0:8000` and accessible over the local network.

### Accessing the API

- **Documentation**: http://localhost:8000/docs
- **Redoc**: http://localhost:8000/redoc

## API Endpoints

### Health Check

- `GET /api/health` - Check if the backend is operational

### Trip Management

- `GET /api/trips` - List available trips for frontend evaluation
- `GET /api/trips/{trip_id}` - Get trip metadata (observation count, dataset availability, map matching status)

### Map Matching

- `POST /api/map-match/{trip_id}` - Execute complete map matching pipeline:
  1. Validate trip ID
  2. Load trip data
  3. Generate road-segment candidates
  4. Engineer features (spatial, temporal, network, transition)
  5. Predict emission probabilities using Random Forest
  6. Run HMM + Viterbi decoding
  7. Save reconstructed route as CSV
  8. Return route and execution statistics

### Route Results

- `GET /api/results/{trip_id}` - Retrieve reconstructed route from saved CSV
- `GET /api/results/{trip_id}/geometry` - Get GeoJSON geometry for map visualization

### Evaluation

- `GET /api/evaluation/{trip_id}` - Get evaluation metrics when reference labels are available

## CORS Configuration

CORS is configured via the `FRONTEND_ORIGIN` environment variable (default: `http://localhost:5173`).

To allow the frontend from another laptop on the local network:

1. Find the backend laptop's LAN IP: `ifconfig | grep "inet " | grep -v 127.0.0.1`
2. Set `FRONTEND_ORIGIN=http://<LAN_IP>:5173` in the `.env` file
3. The frontend should configure its API base URL to `http://<BACKEND_LAN_IP>:8000`

## Map Matching Pipeline

The pipeline executed by `POST /api/map-match/{trip_id}` consists of:

1. **GPS data loading** - Load track data from the dataset
2. **Candidate generation** - Find nearby road segments for each GPS observation
3. **Feature engineering** - Compute spatial, temporal, network, and transition features
4. **Random Forest emission probabilities** - Predict probabilities using the trained classifier
5. **HMM + Viterbi decoding** - Find the most likely sequence of road arcs
6. **Route reconstruction** - Save results and return statistics

## Security

- Reference `.route` files are never exposed through the API
- Ground-truth labels are kept exclusively on the backend
- Trip IDs are validated to prevent path traversal attacks
- Map matching results are separated from evaluation data
- Student-test labels are not used for training or hyperparameter selection

## Troubleshooting

### Common Errors

- **Model not found**: Train the Random Forest model or place it at `outputs/models/random_forest.joblib`
- **Trip not found**: Verify the trip ID exists in the dataset directory
- **No candidates generated**: Increase `GPS_SEARCH_RADIUS` in `.env`
- **Map matching fails**: Check that the dataset files are valid and the model is trained

### Starting from Another Laptop

1. On the backend laptop: start uvicorn as above
2. On the frontend laptop: configure API base URL to the backend's LAN IP
3. Ensure both laptops are on the same local network
4. Use `ifconfig` (macOS) or `ip addr` (Linux) to find the LAN IP