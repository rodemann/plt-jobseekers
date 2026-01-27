import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
import pickle
import json
from datetime import datetime
from typing import Tuple, Dict, Optional, List
import os


# ============================================================================
# DATA PREPARATION
# ============================================================================

def encode_features(
    X: pd.DataFrame,
    categorical_features: List[str],
    ordinal_features: List[str],
    cat_categories: Dict = None,
    ord_mappings: Dict = None
) -> Tuple[pd.DataFrame, Dict, Dict]:
    """
    Encode categorical (one-hot) and ordinal (scaled 0-1) features.
    """
    X_encoded = X.copy()

    # Fit or reuse categorical categories
    if cat_categories is None:
        cat_categories = {col: sorted(X[col].dropna().unique())
                          for col in categorical_features if col in X.columns}

    # Fit or reuse ordinal mappings
    if ord_mappings is None:
        ord_mappings = {}
        for col in ordinal_features:
            if col in X.columns:
                vals = sorted(X[col].dropna().unique())
                n = len(vals)
                ord_mappings[col] = {v: i/(n-1) if n > 1 else 0.0 for i, v in enumerate(vals)}

    # Apply categorical (one-hot, drop first)
    for col, cats in cat_categories.items():
        if col not in X_encoded.columns:
            continue
        for cat in cats[1:]:
            X_encoded[f"{col}_{cat}"] = (X_encoded[col] == cat).astype(float)
        X_encoded = X_encoded.drop(columns=[col])

    # Apply ordinal
    for col, mapping in ord_mappings.items():
        if col not in X_encoded.columns:
            continue
        X_encoded[col] = X_encoded[col].map(lambda v: mapping.get(v, -1.0))

    return X_encoded, cat_categories, ord_mappings


def prepare_features_and_outcome(
    df: pd.DataFrame,
    outcome_col: str,
    categorical_features: List[str] = None,
    ordinal_features: List[str] = None,
    filter_to_unemployed: bool = True,
    cat_categories: Dict = None,
    ord_mappings: Dict = None
) -> Tuple[pd.DataFrame, pd.Series, Dict, Dict]:
    # Filter to unemployed (if specified)
    if filter_to_unemployed:
        df_model = df[df['still_unemployed'] == 1].copy()
    else:
        df_model = df.copy()

    # Separate features and outcome
    feature_cols = [col for col in df_model.columns
                    if col not in ['person_id', outcome_col]]

    X = df_model[feature_cols].copy()
    y = df_model[outcome_col].copy()

    # Apply encoding if features specified
    if categorical_features or ordinal_features:
        X, cat_categories, ord_mappings = encode_features(
            X,
            categorical_features or [],
            ordinal_features or [],
            cat_categories,
            ord_mappings
        )

    return X, y, cat_categories, ord_mappings

# ============================================================================
# MODEL EVALUATION
# ============================================================================

def evaluate_model_on_set(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    set_name: str = "test"
) -> Dict:

    # Get predictions
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    # Calculate metrics
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    auc = roc_auc_score(y, y_proba)
    cm = confusion_matrix(y, y_pred)

    results = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'roc_auc': auc,
        'confusion_matrix': cm.tolist(),
        'n_samples': len(y),
        'n_positive': int(y.sum()),
        'n_negative': int((1-y).sum()),
        'y_true': y,
        'predictions': y_pred,
        'probabilities': y_proba
    }

    return results

def evaluate_model(
    model: LogisticRegression,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: List[str],
    verbose: bool = True
) -> Dict:


    results = {
        'train': evaluate_model_on_set(model, X_train, y_train, "train"),
        'test': evaluate_model_on_set(model, X_test, y_test, "test"),
    }

    return results


# ============================================================================
# SAVE/LOAD MODEL
# ============================================================================

def save_model_and_results(
    model,
    results: Dict,
    feature_names: List[str],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    run_name: str = "model",
    save_dir: str = "models/"
) -> Dict[str, str]:
   
    # Create timestamped subfolder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = f"{timestamp}_{run_name}"
    run_dir = os.path.join(save_dir, run_folder)
    os.makedirs(run_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(run_dir, "model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    results_clean = results.copy()
    # Remove large arrays
    for key in ['train', 'test']:
        if key in results_clean:
            results_clean[key] = {k: v for k, v in results_clean[key].items()
                                  if k not in ['predictions', 'probabilities', 'y_true']}

    results_clean['feature_names'] = feature_names
    results_clean['timestamp'] = timestamp
    results_clean['run_name'] = run_name

    results_path = os.path.join(run_dir, "results.json")
    with open(results_path, 'w') as f:
        json.dump(results_clean, f, indent=2)


    # Save feature domains (all unique values for each feature from both train and test)
    feature_domains = {}
    for col in X_train.columns:
        # Combine unique values from both train and test sets
        train_vals = set(X_train[col].dropna().unique())
        test_vals = set(X_test[col].dropna().unique())
        unique_vals = train_vals.union(test_vals)

        # Convert to native Python types for JSON serialization
        if pd.api.types.is_numeric_dtype(X_train[col]):
            feature_domains[col] = sorted([float(v) for v in unique_vals])
        else:
            feature_domains[col] = sorted([str(v) for v in unique_vals])

    domains_path = os.path.join(run_dir, "feature_domains.json")
    with open(domains_path, 'w') as f:
        json.dump(feature_domains, f, indent=2)


    # Save predictions and probabilities
    test_predictions_df = pd.DataFrame({
        'y_true': results['test']['y_true'].values,
        'y_pred': results['test']['predictions'],
        'y_pred_proba': results['test']['probabilities']
    })
    predictions_path = os.path.join(run_dir, "test_predictions.csv") 
    test_predictions_df.to_csv(predictions_path, index=False)

    train_predictions_df = pd.DataFrame({
        'y_true': results['train']['y_true'].values,
        'y_pred': results['train']['predictions'],
        'y_pred_proba': results['train']['probabilities']
    })
    predictions_path = os.path.join(run_dir, "train_predictions.csv")
    train_predictions_df.to_csv(predictions_path, index=False)


# ============================================================================
# TRAINING AND EVAL PIPELINE
# ============================================================================

def train_and_evaluate(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    outcome_col: str,
    categorical_features: List[str] = None,
    ordinal_features: List[str] = None,
    filter_train_to_unemployed: bool = False,
    filter_test_to_unemployed: bool = True,
    run_name: str = "model",
    save_dir: str = "results/",
    random_state: int = 42
):

    # Prepare training data (fits encoding)
    X_train, y_train, cat_categories, ord_mappings = prepare_features_and_outcome(
        train_df, outcome_col,
        categorical_features=categorical_features,
        ordinal_features=ordinal_features,
        filter_to_unemployed=filter_train_to_unemployed
    )
    feature_names = list(X_train.columns)

    # Prepare test data (uses fitted encoding)
    X_test, y_test, _, _ = prepare_features_and_outcome(
        test_df, outcome_col,
        categorical_features=categorical_features,
        ordinal_features=ordinal_features,
        filter_to_unemployed=filter_test_to_unemployed,
        cat_categories=cat_categories,
        ord_mappings=ord_mappings
    )

    print(f"Train samples: {len(X_train):,}")
    print(f"Features: {len(feature_names)}")
    print(f"Test samples: {len(X_test):,}")

    # Train model
    model = LogisticRegression(
        random_state=random_state,
        solver='lbfgs',
        penalty='l2',
        class_weight=None,
        C=1.0,
        max_iter=1000
    )

    model.fit(X_train, y_train)
    
    
    # Evaluate
    results = evaluate_model(
        model, X_train, y_train, X_test, y_test,
        feature_names
    )

    # Add dataset sizes to results
    results['train_size'] = len(X_train)
    results['test_size'] = len(X_test)

    # Save
    save_model_and_results(
        model=model,
        results=results,
        feature_names=feature_names,
        X_train=X_train,
        X_test=X_test,
        run_name=run_name,
        save_dir=save_dir
    )

    print(f"Train accuracy: {results['train']['accuracy']:.4f}")
    print(f"Train ROC AUC:   {results['train']['roc_auc']:.4f}")
    
    print(f"Test accuracy:  {results['test']['accuracy']:.4f}")
    print(f"Test ROC AUC:   {results['test']['roc_auc']:.4f}")


    return model, results