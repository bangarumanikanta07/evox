from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

class AnomalyService:
    """
    Genuine Scikit-learn IsolationForest anomaly detection engine.
    Trains IsolationForest on the numerical feature matrix and projects records
    onto 2D coordinates using PCA for visual radar scatter.
    """
    def detect_anomalies(self, df: pd.DataFrame, target_col: Optional[str] = None, contamination: float = 0.05) -> Dict[str, Any]:
        # Filter numerical features
        feature_df = df.drop(columns=[target_col]) if target_col and target_col in df.columns else df
        num_df = feature_df.select_dtypes(include=[np.number])

        if num_df.empty or num_df.shape[1] == 0:
            # Fallback to integer representation if all categorical
            num_df = pd.DataFrame({col: pd.factorize(feature_df[col])[0] for col in feature_df.columns})

        # Impute missing and standardize
        imputer = SimpleImputer(strategy="median")
        X_imputed = imputer.fit_transform(num_df)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)

        # Genuine Isolation Forest training
        iso = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100,
            n_jobs=-1,
        )
        preds = iso.fit_predict(X_scaled)  # -1 for outliers, 1 for inliers
        raw_scores = iso.decision_function(X_scaled)  # Negative values are outliers

        # 2D PCA projection for radar plot
        n_components = min(2, X_scaled.shape[1])
        if n_components >= 2:
            pca = PCA(n_components=2, random_state=42)
            coords_2d = pca.fit_transform(X_scaled)
        else:
            coords_2d = np.hstack([X_scaled, np.zeros((len(X_scaled), 1))])

        # Feature standard deviations for root-cause diagnosis
        feature_means = np.mean(X_imputed, axis=0)
        feature_stds = np.std(X_imputed, axis=0)
        feature_stds[feature_stds == 0] = 1.0

        records: List[Dict[str, Any]] = []
        total_records = len(df)
        anomaly_count = int(np.sum(preds == -1))
        normal_count = total_records - anomaly_count

        for i in range(total_records):
            is_anomaly = bool(preds[i] == -1)
            score = float(raw_scores[i])

            # Calculate top deviation factors
            deviations = []
            if is_anomaly:
                row_vals = X_imputed[i]
                z_scores = np.abs((row_vals - feature_means) / feature_stds)
                top_indices = np.argsort(z_scores)[::-1][:2]
                col_names = list(num_df.columns)
                for t_idx in top_indices:
                    deviations.append({
                        "feature": col_names[t_idx],
                        "deviation": round(float(z_scores[t_idx]), 2),
                    })

            record_id = df.index[i]
            if "customer_id" in df.columns:
                record_id = str(df["customer_id"].iloc[i])
            elif "id" in df.columns:
                record_id = str(df["id"].iloc[i])
            else:
                record_id = f"REC_{1000 + i}"

            # Raw feature values snapshot
            row_dict = df.iloc[i].replace({np.nan: None}).to_dict()

            records.append({
                "id": record_id,
                "recordIndex": i,
                "features": row_dict,
                "anomalyScore": round(score, 4),
                "isAnomaly": is_anomaly,
                "projectedX": round(float(coords_2d[i, 0]), 2),
                "projectedY": round(float(coords_2d[i, 1]), 2),
                "topContributingFactors": deviations,
            })

        anomaly_rate = round((anomaly_count / max(1, total_records)) * 100, 2)

        return {
            "totalRecords": total_records,
            "normalCount": normal_count,
            "anomalyCount": anomaly_count,
            "anomalyRate": anomaly_rate,
            "threshold": round(float(np.percentile(raw_scores, contamination * 100)), 4),
            "anomalies": records,
        }

anomaly_service = AnomalyService()
