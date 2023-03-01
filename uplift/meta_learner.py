from functools import partial
import numpy as np
import pandas as pd

import optuna.integration.lightgbm as lgb

from lightgbm import early_stopping
from lightgbm import LGBMClassifier


class XLearner:
    """
    Net Value Optimized X Learner

    Uplift Modeling for Multiple Treatments with Cost Optimization
    Zhao & Harinen (2019)

    Parameters:
    ----------------
    ic_lookup (dict): impresson cost lookup for each treatment
    cc_lookup (dict): conversion cost lookup for each treatment
    control (int): key for control in lookups
    base_classifier partial(Classifier): classifier used for any task in
        the optimziation pipeline for classification. Any sklearn complient
        classifier will work here.
    base_regressor partial(Regressor): regressor used for any task in
        the optimziation pipeline for regression. Any sklearn compliant
        regressor will work here.
    """

    def __init__(self, ic_lookup=None, cc_lookup=None, control=0):
        self.control = control
        self.ic_lookup = ic_lookup
        self.cc_lookup = cc_lookup

    def make_valid_sets(self, X, y, frac=0.1):
        idx = np.arange(X.shape[0])
        val_idx = np.random.choice(idx, int(X.shape[0] * frac))
        dtrain = lgb.Dataset(X[~val_idx], label=y[~val_idx])
        dval = lgb.Dataset(X[val_idx], label=y[val_idx])
        return dtrain, dval

    def fit(self, X, y, T, v):
        """
        X (num_samples, num_features): feature array for observations
        y (num_samples, 1): boolean outcome
        T (num_samples, 1): observed treatments
        v (num_samples, 1): estimated value of conversion
        """

        self.treatments_ = np.unique(T)
        self.response_models = {}
        self.psuedo_models = {}

        # stage 1: fit response_models
        for t in self.treatments_:
            treatment_idx = np.where(T == t)[0]
            dtrain, dval = self.make_valid_sets(X[treatment_idx], y[treatment_idx])
            m = lgb.train(
                {
                    "objective": "binary",
                    "boosting_type": "gbdt",
                },
                dtrain,
                valid_sets=[dtrain, dval],
                callbacks=[early_stopping(100)],
            )
            self.response_models[t] = m

        # stage 2: Compute psuedo treatment effects
        for t in self.treatments_:
            mj = self.response_models[t]
            m0 = self.response_models[self.control]

            treatment_idx = np.where(T == t)[0]
            control_idx = np.where(T == self.control)[0]
            if self.cc_lookup is not None and self.ic_lookup is not None:
                s_t0 = np.ones(control_idx.shape[0]) * self.cc_lookup[self.control]
                s_tj = np.ones(treatment_idx.shape[0]) * self.cc_lookup[t]

                ic_t0 = np.ones(control_idx.shape[0]) * self.ic_lookup[self.control]
                ic_tj = np.ones(treatment_idx.shape[0]) * self.cc_lookup[t]

                D_t0 = (v[control_idx] - s_t0) * (
                    (v[control_idx] - s_t0) * mj.predict(X[control_idx])
                    - v[control_idx] * y[control_idx]
                ) - ic_t0

                D_tj = (v[treatment_idx] - s_tj) * (
                    y[treatment_idx] - v[treatment_idx] * m0.predict(X[treatment_idx])
                ) - ic_tj
            else:
                D_t0 = mj.predict(X[control_idx]) - y[control_idx]
                D_tj = y[treatment_idx] - m0.predict(X[treatment_idx])

            t0_effect_key = f"{self.control}_{t}"
            tj_effect_key = f"{t}_{self.control}"

            dtrain_control, dval_control = self.make_valid_sets(X[control_idx], D_t0)
            dtrain_treatment, dval_treatment = self.make_valid_sets(
                X[treatment_idx], D_tj
            )
            self.psuedo_models[t0_effect_key] = lgb.train(
                {"objective": "regression", "metric": "l2"},
                dtrain_control,
                valid_sets=[dtrain_control, dval_control],
                callbacks=[early_stopping(100)],
            )
            self.psuedo_models[tj_effect_key] = lgb.train(
                {"objective": "regression", "metric": "l2"},
                dtrain_treatment,
                valid_sets=[dtrain_treatment, dval_treatment],
                callbacks=[early_stopping(100)],
            )

        # stage 3: Fit propensity model
        self.e_ = LGBMClassifier().fit(X, T)

    def predict_cate(self, X):
        """
        output CATE for each treatment
        Assumes 0 CATE for control
        """
        cate_tracker = {}
        probs = self.e_.predict_proba(X)
        for t in self.treatments_:
            if t == self.control:
                continue
            tau0_key = f"{self.control}_{t}"
            tauj_key = f"{t}_{self.control}"

            denom = probs[:, self.control] + probs[:, t]
            cate = probs[:, t] / denom * self.psuedo_models[tau0_key].predict(
                X
            ) + probs[:, self.control] * self.psuedo_models[tauj_key].predict(X)
            cate_tracker[t] = cate

        cate_tracker[self.control] = 0
        return pd.DataFrame(cate_tracker)

    def get_best_action(self, X):
        cate = self.predict_cate(X)
        best_action = cate.idxmax(axis=1)
        return best_action
