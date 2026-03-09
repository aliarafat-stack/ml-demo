import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "Telco-Customer-Churn.csv")

CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]

NUMERIC_COLS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]


@st.cache_data
def load_raw_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    return df


@st.cache_data
def get_encoded_data() -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """Return encoded DataFrame and dict of fitted LabelEncoders (for inverse transforms)."""
    df = load_raw_data().copy()
    encoders: dict[str, LabelEncoder] = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, encoders


@st.cache_data
def get_train_test(test_size: float = 0.2, random_state: int = 42):
    df, encoders = get_encoded_data()
    feature_cols = CATEGORICAL_COLS + NUMERIC_COLS
    X = df[feature_cols]
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test, encoders, feature_cols


@st.cache_data
def get_onehot_train_test(test_size: float = 0.2, random_state: int = 42):
    """One-Hot Encoded data for Logistic Regression. Same split indices as get_train_test."""
    df = load_raw_data().copy()
    df_oh = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)
    feature_cols_oh = [c for c in df_oh.columns if c not in ["customerID", "Churn"]]
    X = df_oh[feature_cols_oh]
    y = df_oh["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test, feature_cols_oh


@st.cache_data
def get_scaled_train_test(test_size: float = 0.2, random_state: int = 42):
    """Return scaled features (needed for SGDClassifier / Logistic Regression)."""
    X_train, X_test, y_train, y_test, encoders, feature_cols = get_train_test(
        test_size, random_state
    )
    scaler = StandardScaler()
    X_train_sc = pd.DataFrame(
        scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index
    )
    X_test_sc = pd.DataFrame(
        scaler.transform(X_test), columns=feature_cols, index=X_test.index
    )
    return X_train_sc, X_test_sc, y_train, y_test, encoders, feature_cols, scaler
