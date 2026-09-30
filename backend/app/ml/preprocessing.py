from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

class Preprocessor:
    """
    Genuine Scikit-learn ColumnTransformer and Pipeline.
    Supports numerical scaling, categorical one-hot encoding, and missing value imputation.
    """
    def __init__(self, target_col: str, problem_type: str = "classification"):
        self.target_col = target_col
        self.problem_type = problem_type
        self.transformer: Optional[ColumnTransformer] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.feature_names: List[str] = []
        self.num_cols: List[str] = []
        self.cat_cols: List[str] = []

    def fit_transform(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        # Separate features and target
        if self.target_col not in df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in dataset columns: {list(df.columns)}")

        X_df = df.drop(columns=[self.target_col])
        y_series = df[self.target_col]

        # Identify numerical and categorical features
        self.num_cols = list(X_df.select_dtypes(include=[np.number]).columns)
        self.cat_cols = list(X_df.select_dtypes(exclude=[np.number]).columns)

        transformers = []
        if self.num_cols:
            num_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ])
            transformers.append(("num", num_pipe, self.num_cols))

        if self.cat_cols:
            cat_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ])
            transformers.append(("cat", cat_pipe, self.cat_cols))

        self.transformer = ColumnTransformer(transformers=transformers, remainder="drop")
        X_trans = self.transformer.fit_transform(X_df)

        # Build feature names output
        self.feature_names = []
        if self.num_cols:
            self.feature_names.extend(self.num_cols)
        if self.cat_cols:
            try:
                onehot = self.transformer.named_transformers_["cat"].named_steps["onehot"]
                encoded_cats = onehot.get_feature_names_out(self.cat_cols)
                self.feature_names.extend(list(encoded_cats))
            except Exception:
                self.feature_names.extend(self.cat_cols)

        # Process target
        if self.problem_type == "classification":
            self.label_encoder = LabelEncoder()
            y_trans = self.label_encoder.fit_transform(y_series.astype(str))
        else:
            y_trans = pd.to_numeric(y_series, errors="coerce").fillna(0.0).to_numpy()

        # Stratified train/test split (80/20)
        stratify = y_trans if (self.problem_type == "classification" and len(np.unique(y_trans)) < 15 and np.min(np.bincount(y_trans)) >= 2) else None
        X_train, X_test, y_train, y_test = train_test_split(
            X_trans, y_trans, test_size=0.2, random_state=42, stratify=stratify
        )

        return X_train, X_test, y_train, y_test, self.feature_names

    def transform_input(self, input_dict: Dict[str, Any]) -> np.ndarray:
        """Transforms a single feature dictionary for prediction inference."""
        if self.transformer is None:
            raise ValueError("Preprocessor has not been fitted.")

        input_df = pd.DataFrame([input_dict])
        # Ensure all columns present
        for col in self.num_cols + self.cat_cols:
            if col not in input_df.columns:
                input_df[col] = np.nan

        return self.transformer.transform(input_df)
