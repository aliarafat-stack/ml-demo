import streamlit as st
import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss,
)
import shap


@st.cache_resource
def train_tree_models(_X_train, _y_train):
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            random_state=42, eval_metric="logloss",
        ),
    }
    for model in models.values():
        model.fit(_X_train, _y_train)
    return models


@st.cache_resource
def train_lr_model(_X_train_oh, _y_train):
    lr = LogisticRegression(max_iter=2000, random_state=42)
    lr.fit(_X_train_oh, _y_train)
    return lr



def evaluate_model(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = model.decision_function(X_test)
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0),
        "AUC": roc_auc_score(y_test, y_proba),
    }


def evaluate_all_models(models: dict, X_test, y_test) -> dict:
    return {name: evaluate_model(m, X_test, y_test) for name, m in models.items()}


@st.cache_resource
def get_shap_explainer(_model, _X_train):
    explainer = shap.TreeExplainer(_model)
    shap_values = explainer(_X_train)
    return explainer, shap_values


def get_shap_single(explainer, X_single):
    return explainer(X_single)


def create_sgd_model(classes=None):
    model = SGDClassifier(
        loss="log_loss", penalty="l2", alpha=0.0001,
        random_state=42, warm_start=False,
    )
    if classes is not None:
        model.partial_fit(
            np.zeros((1, 1)), np.array([classes[0]]), classes=classes
        )
    return model


def evaluate_streaming(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0),
        "Log Loss": log_loss(y_test, y_proba),
    }
