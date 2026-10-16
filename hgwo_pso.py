"""
Universal Dataset-Independent Implementation of:
Hybrid Grey Wolf Optimizer - Particle Swarm Optimization (HGWO-PSO)
for Safety-Critical Feature Selection & Classifier Hyperparameter Co-Tuning.

Suitable for rigorous research and testing across arbitrary datasets.
"""

import numpy as np
import pandas as pd
import time
import warnings
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')

class HGWO_PSO(BaseEstimator, ClassifierMixin):
    """
    Hybrid Grey Wolf Optimizer - Particle Swarm Optimization (HGWO-PSO)
    Wrapper Feature Selection and Hyperparameter Co-Tuning.
    
    Parameters
    ----------
    n_wolves : int, default=20
        Number of search agents (population size).
    max_iter : int, default=20
        Number of evolutionary iterations.
    cv_splits : int, default=5
        Number of folds for Stratified Cross-Validation.
    critical_class : int or str, default=0
        The label corresponding to the safety-critical / minority fault class.
    feature_bounds : tuple (min_ratio, max_ratio), default=(0.20, 0.50)
        Preferred ratio of selected features to encourage hardware compression.
    weight_acc : float, default=0.5
        Weight of Classification Accuracy in multi-objective fitness.
    weight_f1 : float, default=0.5
        Weight of Macro-F1 score in multi-objective fitness.
    weight_mdr : float, default=0.35
        Penalty weight for Critical Miss Detection Rate (MDR).
    seed : int, default=42
        Random seed for reproducibility.
    """
    def __init__(self, n_wolves=20, max_iter=20, cv_splits=5, critical_class=0,
                 feature_bounds=(0.20, 0.50), weight_acc=0.5, weight_f1=0.5,
                 weight_mdr=0.35, seed=42):
        self.n_wolves = n_wolves
        self.max_iter = max_iter
        self.cv_splits = cv_splits
        self.critical_class = critical_class
        self.feature_bounds = feature_bounds
        self.weight_acc = weight_acc
        self.weight_f1 = weight_f1
        self.weight_mdr = weight_mdr
        self.seed = seed

    def _calc_mdr(self, y_true, y_pred):
        """Calculates Miss Detection Rate for the critical class."""
        true_crit = (y_true == self.critical_class)
        pred_non_crit = (y_pred != self.critical_class)
        missed = np.sum(true_crit & pred_non_crit)
        total_crit = np.sum(true_crit)
        return float(missed / total_crit) if total_crit > 0 else 0.0

    def _evaluate(self, X, y, particle, num_features):
        mask = (particle[:num_features] > 0.5).astype(int)
        k = np.sum(mask)
        if k == 0:
            return 999.0, 0.0, 0.0, 1.0, 0

        # Dataset-independent feature compression penalty using bounds
        min_k = max(1, int(self.feature_bounds[0] * num_features))
        max_k = max(min_k, int(self.feature_bounds[1] * num_features))
        
        sensor_penalty = 0.0
        if k < min_k:
            sensor_penalty = (min_k - k) * (0.1 / num_features)
        elif k > max_k:
            sensor_penalty = (k - max_k) * (0.1 / num_features)

        # Dynamic mapping for Random Forest hyperparameters
        n_est = int(np.clip(particle[num_features] * 80 + 40, 40, 120))
        max_d = int(np.clip(particle[num_features + 1] * 6 + 4, 4, 10))
        w_crit = float(np.clip(particle[num_features + 2] * 3.0 + 1.5, 1.5, 4.5))

        X_sel = X[:, mask.astype(bool)]
        skf = StratifiedKFold(n_splits=self.cv_splits, shuffle=True, random_state=self.seed)
        
        accs, f1s, mdrs = [], [], []
        cw = {self.critical_class: w_crit}
        # Balance remaining classes
        unique_classes = np.unique(y)
        for c in unique_classes:
            if c != self.critical_class:
                cw[c] = 1.0
        
        for tr, te in skf.split(X_sel, y):
            clf = RandomForestClassifier(n_estimators=n_est, max_depth=max_d,
                                         class_weight=cw, random_state=self.seed, n_jobs=1)
            clf.fit(X_sel[tr], y[tr])
            preds = clf.predict(X_sel[te])
            accs.append(accuracy_score(y[te], preds))
            f1s.append(f1_score(y[te], preds, average="macro"))
            mdrs.append(self._calc_mdr(y[te], preds))

        avg_acc = float(np.mean(accs))
        avg_f1 = float(np.mean(f1s))
        avg_mdr = float(np.mean(mdrs))

        # Scientific multi-objective fitness
        fitness = -(self.weight_acc * avg_acc + self.weight_f1 * avg_f1) + \
                  self.weight_mdr * avg_mdr + 0.05 * (k / num_features) + sensor_penalty
        return fitness, avg_acc, avg_f1, avg_mdr, k

    def fit(self, X, y, feature_names=None):
        """
        Dynamically optimizes feature mask and model hyperparameters from scratch.
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.tolist()
            X_arr = X.values
        else:
            X_arr = np.array(X)
            self.feature_names_in_ = feature_names or [f"Feature_{i}" for i in range(X_arr.shape[1])]

        y_arr = np.array(y)
        num_features = X_arr.shape[1]
        dims = num_features + 3 # Features + n_estimators + max_depth + crit_weight

        np.random.seed(self.seed)
        pos = np.random.uniform(0.0, 1.0, size=(self.n_wolves, dims))
        vel = np.random.uniform(-0.5, 0.5, size=(self.n_wolves, dims))

        alpha_pos = np.zeros(dims)
        alpha_fit = np.inf
        alpha_stats = (0, 0, 1.0, 0)
        
        beta_pos, beta_fit = np.zeros(dims), np.inf
        delta_pos, delta_fit = np.zeros(dims), np.inf
        
        pbest_pos = pos.copy()
        pbest_fit = np.full(self.n_wolves, np.inf)
        self.history_ = []

        start_time = time.time()
        for it in range(self.max_iter):
            a = 2.0 - 2.0 * (it / self.max_iter)
            w = 0.9 - 0.5 * (it / self.max_iter)

            for i in range(self.n_wolves):
                fit, acc, f1, mdr, k = self._evaluate(X_arr, y_arr, pos[i], num_features)
                
                if fit < pbest_fit[i]:
                    pbest_fit[i] = fit
                    pbest_pos[i] = pos[i].copy()

                if fit < alpha_fit:
                    delta_fit, delta_pos = beta_fit, beta_pos.copy()
                    beta_fit, beta_pos = alpha_fit, alpha_pos.copy()
                    alpha_fit, alpha_pos = fit, pos[i].copy()
                    alpha_stats = (acc, f1, mdr, k)
                elif fit < beta_fit:
                    delta_fit, delta_pos = beta_fit, beta_pos.copy()
                    beta_fit, beta_pos = fit, pos[i].copy()
                elif fit < delta_fit:
                    delta_fit, delta_pos = fit, pos[i].copy()

            for i in range(self.n_wolves):
                r1, r2, r3 = np.random.rand(dims), np.random.rand(dims), np.random.rand(dims)
                
                # GWO leader guidance
                D_alpha = np.abs(2 * r1 * alpha_pos - pos[i])
                X1 = alpha_pos - (2 * a * np.random.rand(dims) - a) * D_alpha
                
                D_beta = np.abs(2 * r2 * beta_pos - pos[i])
                X2 = beta_pos - (2 * a * np.random.rand(dims) - a) * D_beta
                
                D_delta = np.abs(2 * r3 * delta_pos - pos[i])
                X3 = delta_pos - (2 * a * np.random.rand(dims) - a) * D_delta
                
                X_gwo = (X1 + X2 + X3) / 3.0
                vel[i] = w * vel[i] + 1.5 * r1 * (X1 - pos[i]) + 1.5 * r2 * (X2 - pos[i]) + 1.5 * r3 * (X3 - pos[i])
                pos[i] = np.clip(0.5 * (pos[i] + vel[i]) + 0.5 * X_gwo, 0.0, 1.0)

            self.history_.append({
                "iteration": it + 1,
                "best_fitness": alpha_fit,
                "accuracy": alpha_stats[0],
                "macro_f1": alpha_stats[1],
                "critical_mdr": alpha_stats[2],
                "selected_features_count": alpha_stats[3]
            })

        self.elapsed_time_ = time.time() - start_time
        self.support_ = (alpha_pos[:num_features] > 0.5)
        self.selected_features_ = [self.feature_names_in_[j] for j in range(num_features) if self.support_[j]]
        
        self.best_params_ = {
            "n_estimators": int(np.clip(alpha_pos[num_features] * 80 + 40, 40, 120)),
            "max_depth": int(np.clip(alpha_pos[num_features + 1] * 6 + 4, 4, 10)),
            "crit_weight": float(np.clip(alpha_pos[num_features + 2] * 3.0 + 1.5, 1.5, 4.5))
        }
        
        self.best_score_ = {
            "accuracy": alpha_stats[0],
            "macro_f1": alpha_stats[1],
            "critical_mdr": alpha_stats[2],
            "selected_features": len(self.selected_features_)
        }

        # Train final estimator on full selected dataset
        cw = {self.critical_class: self.best_params_["crit_weight"]}
        for c in np.unique(y_arr):
            if c != self.critical_class:
                cw[c] = 1.0
                
        self.estimator_ = RandomForestClassifier(
            n_estimators=self.best_params_["n_estimators"],
            max_depth=self.best_params_["max_depth"],
            class_weight=cw,
            random_state=self.seed
        )
        self.estimator_.fit(X_arr[:, self.support_], y_arr)
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X[self.selected_features_]
        return np.array(X)[:, self.support_]

    def predict(self, X):
        X_trans = self.transform(X)
        return self.estimator_.predict(X_trans)

    def predict_proba(self, X):
        X_trans = self.transform(X)
        return self.estimator_.predict_proba(X_trans)


def load_dataset(csv_path, target_col="EventFlag", drop_cols=None, critical_val="Runaway"):
    """
    Universal dataset preprocessor that dynamically extracts features,
    encodes strings, scales numerics, and maps the critical safety target.
    Works for any tabular dataset.
    """
    df = pd.read_csv(csv_path)
    
    # 1. Target mapping: critical vs non-critical
    if critical_val is not None and critical_val in df[target_col].values:
        # Standard 3-class safety mapping if intermediate warnings exist
        y = np.zeros(len(df), dtype=int)
        for i, val in enumerate(df[target_col].values):
            if val == critical_val:
                y[i] = 0 # Critical
            elif str(val).lower() in ['watch', 'ok', 'normal']:
                y[i] = 1 # Normal
            else:
                y[i] = 2 # Warning / Elevated
    else:
        y = LabelEncoder().fit_transform(df[target_col])

    # 2. Drop user-specified or identifier columns
    if drop_cols is None:
        drop_cols = [c for c in df.columns if 'id' in c.lower() or 'time' in c.lower() or 'note' in c.lower() or 'status' in c.lower()]
    drop_cols.append(target_col)
    
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    # 3. Dynamic categorical encoding
    for col in X.select_dtypes(include=['object', 'category', 'bool']).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        
    feature_names = X.columns.tolist()
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_names)
    return X_scaled, y, feature_names
