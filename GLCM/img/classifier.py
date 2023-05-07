#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import argparse

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import lightgbm as lgb
#from lightgbm import plot_importance, plot_metric


class Classfier():

    def __init__(self, csv_file:str, seed:int=44):
        names = ("csv_file", "type", 
                 "DSS1", "DSS3", "DSS5",
                 "COR1", "COR3", "COR5",
                 "ENG1", "ENG3", "ENG5",
                 "HMG1", "HMG3", "HMG5",
                 "CTR1", "CTR3", "CTR5",
                 "ENT1", "ENT3", "ENT5")
        self.data = pd.read_csv(csv_file, names=names)
        self.seed = seed
        self.encoder = LabelEncoder()
        self.model = None


    def preprocess_data(self):
        self.data['type'] = self.encoder.fit_transform(self.data['type'])
        self.train_data, self.test_data =\
            train_test_split(self.data,
                             test_size=0.2,
                             random_state=self.seed)


    def train(self) -> None:
        # Make training data
        X_train = self.train_data.drop(columns=['csv_file', 'type'])
        y_train = self.train_data['type']
        lgb_train = lgb.Dataset(X_train, y_train)

        # Add evaluation data for plot_metric()
        X_test = self.test_data.drop(columns=['csv_file', 'type'])
        y_test = self.test_data['type']
        lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

        params = {
            'objective': 'multiclass',
            'num_class': len(self.encoder.classes_),
            'metric': 'multi_logloss',
            'verbosity': 1,
            'max_depth': 6,
            'num_leaves': 31,
            'min_data_in_leaf': 20,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'lambda_l1': 0.2,
            'lambda_l2': 0.2,
        }

        # Record metrics during training
        evals_result = {}
        self.model = lgb.train(
            params,
            lgb_train,
            valid_sets=[lgb_train, lgb_eval],
            evals_result=evals_result
        )

        self.evals_result = evals_result
        #return evals_result


    def predict(self, data):
        return self.encoder.inverse_transform(
            self.model.predict(data, num_iteration=self.model.best_iteration)
        )


    def evaluate(self) -> float:
        X_test = self.test_data.drop(columns=['csv_file', 'type'])
        y_test = self.test_data['type']
        y_pred = self.model.predict(X_test).argmax(axis=1)
        #y_true = self.encoder.inverse_transform(y_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy
        #return classification_report(y_test, y_pred)
        #return classification_report(y_true, y_pred)


    def show_summary(self):
        # show the summary of trainning
        print(f"Training Summary:")
        print(f"Training data size: {len(self.train_data)}")
        print(f"Test data size: {len(self.test_data)}")
        print(f"Number of classes: {len(self.encoder.classes_)}")

        # show the sum
        accuracy = self.evaluate()
        y_test = self.test_data['type']
        y_pred = self.model.predict(
            self.test_data.drop(columns=['csv_file', 'type'])
        ).argmax(axis=1)
        print(f'\nAccuracy: {accuracy:.2f}')
        print(f'\nClassification Report:')
        print(classification_report(y_test, y_pred))
        print(f'\nConfusion Matrix:')
        print(confusion_matrix(y_test, y_pred))

        # show the metrics plot
        print(f'\nMetrics Plot:')
        ax = lgb.plot_metric(self.evals_result)
        plt.show()

        # show the importance plot
        print(f'\nFeature Importance Plot:')
        ax = lgb.plot_importance(self.model)
        plt.show()

        plt.close()


    def save_model(self, model_filename='lightgbm_texture.txt'):
        self.model.save_model(model_filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="ML model for Mark classification"
    )
    parser.add_argument(
        '-i',
        '--input_file',
        dest='input_file',
        type=str,
        help='Path to the CSV file',
    )
    parser.add_argument(
        '-o',
        '-output_file',
        dest='output_file',
        type=str,
        help='Path to the output (model) file',
    )
    args = parser.parse_args()

    ml_model = Classfier(args.input_file)
    ml_model.preprocess_data()
    ml_model.train()
    ml_model.show_summary()
    ml_model.save_model(model_filename=args.output_file)

