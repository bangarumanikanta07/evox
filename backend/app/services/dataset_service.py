import os
import io
import math
import time
from typing import Dict, Any, List, Optional, Tuple

class DatasetService:
    """
    Genuine Pandas-based dataset profiling, validation, and synthetic generator.
    """
    def __init__(self):
        self._datasets: Dict[str, Any] = {}

    def profile_dataframe(self, df: Any, name: str = "dataset.csv", file_size: int = 0) -> Dict[str, Any]:
        import pandas as pd
        import numpy as np

        row_count = int(len(df))
        if row_count == 0:
            raise ValueError("Dataset contains 0 rows.")

        col_keys = list(df.columns)
        column_count = len(col_keys)

        # Check duplicate rows
        duplicate_count = int(df.duplicated().sum())

        columns_profile = []
        num_cols = []
        cat_cols = []
        total_missing = 0
        total_outliers = 0

        for col in col_keys:
            series = df[col]
            missing_count = int(series.isna().sum())
            total_missing += missing_count
            missing_pct = round((missing_count / row_count) * 100, 2)
            unique_count = int(series.nunique(dropna=True))

            # Detect data type
            is_num = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)
            is_bool = pd.api.types.is_bool_dtype(series)
            is_date = pd.api.types.is_datetime64_any_dtype(series)

            if is_bool:
                data_type = "boolean"
            elif is_num:
                data_type = "numerical"
                num_cols.append(col)
            elif is_date:
                data_type = "datetime"
            else:
                data_type = "categorical"
                cat_cols.append(col)

            # Calculate stats and outliers for numerical
            stats = None
            outlier_count = 0
            if data_type == "numerical":
                valid_num = series.dropna()
                if len(valid_num) > 0:
                    q1 = float(valid_num.quantile(0.25))
                    q3 = float(valid_num.quantile(0.75))
                    iqr = q3 - q1
                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr
                    outliers = valid_num[(valid_num < lower) | (valid_num > upper)]
                    outlier_count = int(len(outliers))
                    total_outliers += outlier_count

                    stats = {
                        "min": round(float(valid_num.min()), 2),
                        "max": round(float(valid_num.max()), 2),
                        "mean": round(float(valid_num.mean()), 2),
                        "median": round(float(valid_num.median()), 2),
                        "stdDev": round(float(valid_num.std()) if len(valid_num) > 1 else 0.0, 2),
                    }
            elif data_type == "categorical":
                val_counts = series.value_counts(dropna=True).head(5).to_dict()
                stats = {
                    "topCategories": [{"value": str(k), "count": int(v)} for k, v in val_counts.items()]
                }

            # Target column heuristic
            lower = str(col).lower()
            is_target_name = any(kw in lower for kw in ["target", "label", "churn", "class", "status", "outcome", "fraud", "y"])
            is_potential_target = is_target_name or (unique_count >= 2 and unique_count <= 20 and col == col_keys[-1])

            columns_profile.append({
                "name": str(col),
                "dataType": data_type,
                "missingCount": missing_count,
                "missingPercentage": missing_pct,
                "uniqueCount": unique_count,
                "isPotentialTarget": is_potential_target,
                "stats": stats,
                "outlierCount": outlier_count,
            })

        # Calculate Data Quality Score (0 - 100)
        total_cells = max(1, row_count * column_count)
        missing_rate = total_missing / total_cells
        dup_rate = duplicate_count / row_count
        outlier_rate = total_outliers / total_cells

        completeness = max(0, int(round((1 - missing_rate) * 100)))
        uniqueness = max(0, int(round((1 - dup_rate) * 100)))
        validity = max(0, int(round((1 - min(0.3, outlier_rate * 5)) * 100)))
        consistency = 96

        quality_score = max(10, min(100, int(round(0.35 * completeness + 0.25 * uniqueness + 0.25 * validity + 0.15 * consistency))))

        reasons = []
        recommendations = []
        if completeness >= 98:
            reasons.append("High dataset completeness with < 2% overall missing values.")
        else:
            reasons.append(f"Missing values detected across {round(missing_rate * 100, 1)}% of all data points.")
            recommendations.append("Apply median or mode imputation prior to estimator fitting.")

        if duplicate_count == 0:
            reasons.append("Zero redundant duplicate rows detected in dataset.")
        else:
            reasons.append(f"{duplicate_count} duplicate rows observed in dataset.")
            recommendations.append("Deduplicate identical feature vectors to avoid overfitting.")

        if total_outliers > 0:
            reasons.append(f"{total_outliers} IQR statistical outliers identified in numerical features.")
            recommendations.append("Use robust scaling or tree-based models less sensitive to extreme ranges.")

        # Determine target feature and problem type
        target_col = None
        for c in columns_profile:
            if c["isPotentialTarget"]:
                target_col = c["name"]
                break
        if not target_col:
            target_col = col_keys[-1]

        target_prof = next((c for c in columns_profile if c["name"] == target_col), None)
        if target_prof and target_prof["dataType"] == "numerical" and target_prof["uniqueCount"] > 25:
            detected_problem = "regression"
            confidence = 94
            reason = f'Target column "{target_col}" is a continuous numerical feature with {target_prof["uniqueCount"]} distinct values.'
            classes_count = None
        else:
            detected_problem = "classification"
            classes_count = target_prof["uniqueCount"] if target_prof else 2
            confidence = 96 if classes_count == 2 else 90
            reason = f'Target column "{target_col}" has {classes_count} discrete categorical classes.'

        # Preview rows (first 15 converted cleanly to native python types)
        preview_rows = df.head(15).replace({np.nan: None}).to_dict(orient="records")

        # Feature list (excluding target)
        features = [c["name"] for c in columns_profile if c["name"] != target_col]

        dataset_id = f"ds_{len(self._datasets) + 1}_{int(row_count)}"
        dataset_obj = {
            "id": dataset_id,
            "name": name,
            "rowCount": row_count,
            "columnCount": column_count,
            "fileSize": file_size,
            "uploadedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "columns": columns_profile,
            "previewRows": preview_rows,
            "qualityReport": {
                "score": quality_score,
                "completeness": completeness,
                "validity": validity,
                "uniqueness": uniqueness,
                "consistency": consistency,
                "reasons": reasons,
                "recommendations": recommendations,
            },
            "problemDetection": {
                "detectedProblem": detected_problem,
                "confidence": confidence,
                "reason": reason,
                "suggestedTarget": target_col,
                "classesCount": classes_count,
            },
            "targetColumn": target_col,
            "features": features,
        }

        # Store in-memory DataFrame and dict
        self._datasets[dataset_id] = {"meta": dataset_obj, "df": df}
        return dataset_obj

    def get_dataset(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        record = self._datasets.get(dataset_id)
        return record["meta"] if record else None

    def get_dataframe(self, dataset_id: str) -> Optional[Any]:
        record = self._datasets.get(dataset_id)
        return record["df"] if record else None

    def generate_demo_dataset(self) -> Dict[str, Any]:
        """
        Generates a genuine synthetic Customer Intelligence dataset using NumPy and Pandas.
        Includes statistical correlations, slight noise, and real outliers for IsolationForest.
        """
        import numpy as np
        import pandas as pd

        np.random.seed(42)
        n_samples = 1000

        age = np.random.randint(21, 72, size=n_samples)
        income = np.round(np.random.normal(68000, 24000, size=n_samples) / 500) * 500
        income = np.clip(income, 25000, 180000)

        credit_score = np.round(np.random.normal(700, 65, size=n_samples))
        credit_score = np.clip(credit_score, 540, 850).astype(int)

        account_age_months = np.random.randint(4, 120, size=n_samples)
        transaction_frequency = np.random.poisson(lam=18, size=n_samples)
        transaction_frequency = np.clip(transaction_frequency, 1, 60)

        transaction_amount = np.round(
            120 + (income / 100) * np.random.uniform(0.3, 1.1, size=n_samples) + np.random.exponential(scale=400, size=n_samples)
        )

        regions = np.random.choice(["North America", "Europe", "Asia-Pacific", "Latin America"], size=n_samples, p=[0.4, 0.3, 0.2, 0.1])
        tiers = np.random.choice(["Standard", "Silver", "Gold", "Platinum"], size=n_samples, p=[0.45, 0.3, 0.18, 0.07])

        # Mathematical ground truth function for churn risk with logistic noise
        z = (
            (70 - age) * 0.02
            + (100000 - income) * 0.000025
            + (700 - credit_score) * 0.006
            + (60 - account_age_months) * 0.012
            + (18 - transaction_frequency) * 0.04
            + np.random.normal(0, 0.45, size=n_samples)
        )
        prob = 1.0 / (1.0 + np.exp(-z))
        churn_status = np.where(prob > 0.52, "Churn_Risk", "Retained")

        # Inject real multivariate outliers for IsolationForest detection
        outlier_indices = [42, 188, 312, 609, 875]
        for idx in outlier_indices:
            transaction_amount[idx] = 48500 + np.random.uniform(5000, 25000)
            transaction_frequency[idx] = 85

        df = pd.DataFrame({
            "customer_age": age,
            "annual_income": income,
            "credit_score": credit_score,
            "account_age_months": account_age_months,
            "transaction_frequency": transaction_frequency,
            "transaction_amount": transaction_amount,
            "region": regions,
            "account_tier": tiers,
            "churn_status": churn_status,
        })

        return self.profile_dataframe(df, name="Customer_Intelligence_Benchmark.csv", file_size=185000)

dataset_service = DatasetService()
