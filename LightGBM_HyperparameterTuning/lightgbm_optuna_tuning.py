#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import lightgbm as lgb
import optuna

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from optuna.trial import Trial


class LightGBMOptunaTuning:

    def __init__(self,
                 x_train: np.ndarray,
                 x_val: np.ndarray,
                 y_train: np.ndarray,
                 y_val: np.ndarray,
                 num_trials: int = 100):
        self.x_train = x_train
        self.x_val = x_val
        self.y_train = y_train
        self.y_val = y_val
        self.train_data = lgb.Dataset(self.x_train, label=self.y_train)
        self.val_data = lgb.Dataset(self.x_val, label=self.y_val)
        self.num_trials = num_trials

    def objective(self, trial: Trial) -> float:
        boosting_type = trial.suggest_categorical('boosting_type',
                                                  ['gbdt', 'dart', 'goss'])
        params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'verbosity': -1,
            'boosting_type': boosting_type,
            #'boosting_type': trial.suggest_categorical('boosting_type',
            #                                           ['gbdt', 'dart', 'goss']),
            'num_leaves': trial.suggest_int('num_leaves', 10, 500),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.005, 0.5),
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
            #'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
            #'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'feature_pre_filter': False,
        }

        if boosting_type != 'goss':
            params.update({
                'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            })


        model = lgb.train(params,
                          self.train_data,
                          valid_sets=[self.val_data],
                          early_stopping_rounds=100,
                          verbose_eval=True)
        preds = model.predict(self.x_val, num_iteration=model.best_iteration)
        pred_labels = np.argmax(preds, axis=1)

        accuracy = accuracy_score(self.y_val, pred_labels)

        return accuracy

    def optimize(self) -> None:
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=self.num_trials)

        print('Best trial:')
        trial = study.best_trial
        print('  Value: ', trial.value)
        print('  Params: ')
        for key, value in trial.params.items():
            print('    {}:{}'.format(key, value))


def main():
    # Load Iris dataset
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target

    # Split dataset into train and test
    x_train, x_val, y_train, y_val = train_test_split(x, y,
                                                      test_size=0.3,
                                                      random_state=42)

    # Hyperparameter tuning
    hypertuning = LightGBMOptunaTuning(x_train, x_val, y_train, y_val)
    hypertuning.optimize()


if __name__ == '__main__':
    main()



