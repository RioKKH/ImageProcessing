#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import io
import sys
import argparse
import pickle

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder

from GLCM import GLCM
from classifier import Classfier


class ImageClassifier:

    header = ("csv_file", "type", 
              "DSS1", "DSS3", "DSS5", "COR1", "COR3", "COR5",
              "ENG1", "ENG3", "ENG5", "HMG1", "HMG3", "HMG5", 
              "CTR1", "CTR3", "CTR5", "ENT1", "ENT3", "ENT5")  

    def __init__(self,
                 model_path:str,
                 encoder_path:str) -> None:

        self.glcm = GLCM()
        self.model = lgb.Booster(model_file=model_path)
        with open(encoder_path, 'rb') as f:
            self.encoder = pickle.load(f)


    def preprocess_image(self, image_path: str) -> np.ndarray:
        result = self.glcm.run(image_path, pooling=True, padding_size=0)

        csv_string = ','.join(ImageClassifier.header) + '\n' + result
        self.df = pd.read_csv(io.StringIO(csv_string))


    def classify_image(self) -> str:
        class_probabilities = self.model.predict(
            self.df.drop(columns=['csv_file', 'type'])
        )
        predicted_class_index = np.argmax(class_probabilities)
        predicted_class_label = self.encoder.inverse_transform([predicted_class_index])
        print(predicted_class_label[0])

        return predicted_class_label[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description = "ML model classifier for mark detection"
    )
    parser.add_argument(
        '-m',
        '--model',
        dest='model_path',
        type=str,
        help='Path to the model file',
    )
    parser.add_argument(
        '-e',
        '--encoder',
        dest='encoder_path',
        type=str,
        help='Path to the encoder file',
    )
    parser.add_argument(
        '-i',
        '--input_image',
        dest='input_image',
        type=str,
        help='Path to the image file',
    )
    args = parser.parse_args()

    image_classifier = ImageClassifier(args.model_path, args.encoder_path)
    image_classifier.preprocess_image(args.input_image)
    image_classifier.classify_image()



