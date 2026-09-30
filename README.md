# EVOX — Adaptive AI Intelligence Platform

> **"Adaptive Intelligence. Optimized Decisions."**

EVOX is an intelligent, production-ready machine learning experimentation and multi-objective optimization platform. It autonomously profiles tabular datasets with **Pandas**, performs automated preprocessing with **Scikit-Learn ColumnTransformer and Pipeline**, trains competing candidate models with **Scikit-Learn and XGBoost**, extracts non-dominated 3D Pareto frontiers, executes real hyperparameter optimization with **Optuna**, attributes prediction drivers using **SHAP and Permutation Importance**, detects multivariate anomalies using **Isolation Forest**, persists models with **Joblib**, and synthesizes executive AI intelligence reports.

---

## 1. System Architecture

EVOX is architected as a clean full-stack monorepo where the **Python FastAPI backend is the single source of truth for all machine learning operations**:

```
evox/
│
├── src/                               # React + TypeScript + Vite Frontend
│   ├── components/
│   │   ├── common/                    # NeuralCanvas (WebGL/Canvas), PresentationModeModal
│   │   ├── landing/                   # LandingHero (Visual Pipeline DATA → DECIDE)
│   │   ├── layout/                    # Header (3-Zone Top Bar Contract), Sidebar
│   │   └── views/                     # 10 core views:
│   │       ├── OverviewView.tsx       # Live KPI cards, benchmark charts, progression
│   │       ├── DatasetLabView.tsx     # CSV/XLSX upload, 0-100 quality score, profiling
│   │       ├── AutoMLView.tsx         # Preprocessing pipeline, model training, live progress
│   │       ├── ModelArenaView.tsx     # Multi-metric comparisons, architectural inspection
│   │       ├── OptimizationView.tsx   # 3D Pareto frontier, dynamic weights, Optuna
│   │       ├── ExplainView.tsx        # SHAP attribution, permutation importance, waterfall
│   │       ├── PredictionStudioView.tsx# Dynamic feature inputs, live inference & probabilities
│   │       ├── AnomalyRadarView.tsx   # Isolation Forest 2D radar, outlier diagnostics
│   │       ├── ReportView.tsx         # Executive dossier, JSON/CSV export, Print/PDF
│   │       └── SettingsView.tsx       # Compute settings, PRNG seed, remote backend URL
│   ├── services/
│   │   └── api.ts                     # Unified API client querying FastAPI endpoints
│   ├── types/
│   │   └── index.ts                   # Full TypeScript data contracts
│   └── App.tsx                        # Root router and pipeline state management
│
├── backend/                           # Python FastAPI Machine Learning Backend
│   ├── app/
│   │   ├── main.py                    # FastAPI application & CORS configuration
│   │   ├── api/endpoints.py           # REST endpoints
│   │   ├── models/schemas.py          # Pydantic request/response schemas
│   │   ├── models/registry.py         # Joblib model persistence registry
│   │   ├── ml/preprocessing.py        # Scikit-learn ColumnTransformer & Pipeline
│   │   ├── ml/automl.py               # Real model.fit() & multi-metric evaluation
│   │   ├── optimization/pareto.py     # 3D Pareto non-dominated sorting
│   │   ├── optimization/optuna_engine.py# Real Optuna study & cross-validation
│   │   ├── explainability/shap_engine.py# Real SHAP & Permutation feature attributions
│   │   └── services/
│   │       ├── dataset_service.py     # Real Pandas profiling & synthetic generator
│   │       ├── anomaly_service.py     # Real Scikit-learn IsolationForest & PCA
│   │       └── report_service.py      # Real report compilation & synthesis
│   ├── requirements.txt               # Python ML packages
│   └── Dockerfile                     # Backend container image
│
├── server.ts                          # Express + Vite full-stack server & FastAPI proxy
├── docker-compose.yml                 # Local multi-container orchestration
├── render.yaml                        # Render cloud deployment manifest
├── metadata.json                      # AI Studio capabilities configuration
├── package.json
└── README.md
```

---

## 2. Real Machine Learning Implementations

### A. Real Dataset Pipeline & Profiling (`backend/app/services/dataset_service.py`)
- Real Pandas `read_csv` and `read_excel` ingestion.
- True statistical profiling: missing rates, duplicate rows (`df.duplicated().sum()`), numerical vs. categorical detection, quantiles, and IQR outlier detection ($Q_1 - 1.5 \times \text{IQR}$, $Q_3 + 1.5 \times \text{IQR}$).
- Generates a **0–100 Data Quality Score** based on actual completeness, validity, uniqueness, and consistency.

### B. Real Preprocessing (`backend/app/ml/preprocessing.py`)
- `sklearn.compose.ColumnTransformer` and `sklearn.pipeline.Pipeline`.
- Numerical features: `SimpleImputer(strategy='median')` + `StandardScaler()`.
- Categorical features: `SimpleImputer(strategy='most_frequent')` + `OneHotEncoder(handle_unknown='ignore')`.
- Stratified 80/20 train/test partition using `sklearn.model_selection.train_test_split`.

### C. Real AutoML Training (`backend/app/ml/automl.py`)
- **Classification**:
  - `RandomForestClassifier(n_estimators=100, max_depth=10)`
  - `ExtraTreesClassifier(n_estimators=100, max_depth=10)`
  - `GradientBoostingClassifier(n_estimators=100)`
  - `LogisticRegression(max_iter=1000)`
  - `SVC(kernel="linear", probability=True)`
  - `XGBClassifier` (if installed)
- **Regression**:
  - `RandomForestRegressor()`
  - `ExtraTreesRegressor()`
  - `GradientBoostingRegressor()`
  - `LinearRegression()`
  - `SVR()`
- Every model is **actually fitted with `model.fit(X_train, y_train)`**.
- Predictions generated via **`y_pred = model.predict(X_test)`**.
- True metrics computed:
  - Classification: `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `roc_auc_score`, `confusion_matrix`.
  - Regression: `mean_absolute_error`, `mean_squared_error`, `r2_score`.
- Real execution profiling: `trainingTimeMs` and `inferenceLatencyMs` per sample.
- Models persisted locally with **`joblib.dump`**.

### D. Real Isolation Forest (`backend/app/services/anomaly_service.py`)
- `from sklearn.ensemble import IsolationForest`.
- Fitted on preprocessed numerical feature matrix with configurable contamination parameter.
- Real decision function scoring ($s(x, n)$) and anomaly predictions (-1 vs 1).
- 2D PCA projection via `from sklearn.decomposition import PCA` for the Anomaly Radar plot.
- Feature deviation diagnostics calculated via Z-scores.

### E. Real Explainable AI (`backend/app/explainability/shap_engine.py`)
- `import shap`.
- `shap.TreeExplainer` for ensemble decision trees.
- `from sklearn.inspection import permutation_importance` as robust mathematical fallback.
- Instance explanation decomposes predictions into positive and negative force contributions.
- Explicit labeling as statistical model-based association, non-causal.

### F. Real Optuna Hyperparameter Optimization (`backend/app/optimization/optuna_engine.py`)
- `import optuna`.
- Real Optuna study with 3-fold cross validation (`cross_val_score`).
- Explores `n_estimators`, `max_depth`, `min_samples_split`.
- Returns actual trial histories, validation scores, and latency metrics.

### G. 3D Pareto Frontier (`backend/app/optimization/pareto.py`)
- Evaluates non-dominated solutions across Performance (+), Latency (-), and Complexity (-).
- Dynamic objective weighting enables custom priority rankings.

---

## 3. API Endpoints (FastAPI)

- `GET /api/health` — Backend health check and capability status.
- `GET /api/demo` — Generates genuine synthetic customer intelligence dataset.
- `POST /api/dataset/upload` — Upload CSV or XLSX dataset for Pandas profiling.
- `GET /api/dataset/{id}` — Retrieve profiled dataset metadata.
- `POST /api/dataset/analyze` — Run detailed statistical analysis on dataset.
- `POST /api/train` — Train AutoML candidate models via `model.fit()`.
- `GET /api/train/{id}/status` — Check status of training run.
- `POST /api/optimize` — Execute real Optuna hyperparameter study.
- `GET /api/optimization/{id}` — Retrieve optimization trial history.
- `POST /api/predict` — Perform live inference with fitted model & preprocessor.
- `POST /api/explain` — Compute global SHAP / permutation feature attributions.
- `POST /api/anomaly` — Execute Scikit-learn IsolationForest anomaly detection.
- `GET /api/report/{id}` — Retrieve synthesized intelligence report.
- `POST /api/report/generate` — Compile final executive intelligence dossier.

---

## 4. Local Installation & Setup

### Prerequisites
- Python 3.10+
- Node.js 18+

### Step 1: Install Python Backend Dependencies
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2: Run Python FastAPI Backend
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```
Interactive Swagger Documentation: `http://localhost:8001/docs`

### Step 3: Run Full-Stack Web Application
In a separate terminal at the project root:
```bash
npm install
npm run dev
```
Open **`http://localhost:3000`** in your browser. The Node server automatically proxies `/api/*` requests to the Python FastAPI backend.

---

## 5. Docker Deployment

To launch both the frontend and backend together:
```bash
docker-compose up --build
```
- Frontend UI: `http://localhost:3000`
- Python FastAPI Backend: `http://localhost:8000`

---

## 6. Cloud Deployment

### Deploying Frontend to Vercel
1. Import repository into Vercel.
2. Build Command: `npm run build`
3. Output Directory: `dist`
4. Set Environment Variable: `VITE_API_URL` to your deployed Render backend URL (e.g. `https://evox-backend.onrender.com`).

### Deploying Backend to Render
1. Create a Web Service on Render using the provided `render.yaml`.
2. Root Directory: `backend`
3. Build Command: `pip install -r requirements.txt`
4. Start Command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

---

## 7. License

Apache-2.0 License. Built for advanced machine learning research and enterprise decision optimization.
