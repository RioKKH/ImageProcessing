#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle

import lightgbm as lgb
import matplotlib.pyplot as plt
from onnxmltools import convert_lightgbm
from onnxmltools.utils import save_model


class ShowTree:

    def __init__(self,
                 model_path:str,
                 encoder_path:str) -> None:

        self.model = lgb.Booster(model_file=model_path)
        with open(encoder_path, 'rb') as f:
            self.encoder = pickle.load(f)

    def plot_tree(self, 
                  figsize:tuple=(20,8)):
        for tree_index in range(0, 10):
        #for tree_index in [0, 25, 50, 75, 100, 125, 150, 175, 199]:
            print(f"Plotting tree {tree_index}")
            lgb.plot_tree(self.model, tree_index=tree_index, figsize=figsize)
            plt.title(f"Tree {tree_index}")
            plt.savefig(f"tree_{tree_index}.png")
            plt.close()

    def convert_model(self,
                      fout:str="model.onnx"):
        self.onnx_model = convert_lightgbm(self.model)
        save_model(onnx_model, fout)


