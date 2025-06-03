# src/classical_baseline.py

import numpy as np
import optuna
try:
    from cuml.ensemble import RandomForestClassifier as CuMLRandomForestClassifier
except ImportError:
    from sklearn.ensemble import RandomForestClassifier as CuMLRandomForestClassifier

from sklearn.metrics import roc_auc_score

class RandomForestBaseline:
    def __init__(self):
        self.model = None
        self.is_unsupervised = False

    def fit(self, X, y, trial=None, n_estimators=100, max_depth=None, min_samples_split=2):
        if y is not None:
            self.model = CuMLRandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=42,
                n_streams=1
            )
            self.model.fit(X, y)
            self.is_unsupervised = False
        else:
            self.is_unsupervised = True

    def predict(self, X):
        if self.is_unsupervised:
            return np.zeros(len(X))
        return self.model.predict(X).astype(int)

    def predict_proba(self, X):
        if self.is_unsupervised:
            return np.zeros(len(X))
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importance(self):
        if self.is_unsupervised:
            return np.zeros(self.model.n_features_in_)
        return self.model.feature_importances_

def objective_rf(trial, X_train, y_train, X_test, y_test):
    n_estimators = trial.suggest_int("n_estimators", 50, 500)
    max_depth = trial.suggest_int("max_depth", 5, 50)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)

    model = RandomForestBaseline()
    model.fit(X_train, y_train, trial=trial, n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split)
    
    y_pred_proba = model.predict_proba(X_test)
    auc = roc_auc_score(y_test, y_pred_proba)
    return auc

def tune_rf_model(X_train, y_train, X_test, y_test):
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective_rf(trial, X_train, y_train, X_test, y_test), n_trials=30)
    return study.best_params