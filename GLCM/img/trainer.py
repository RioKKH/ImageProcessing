#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm

import GLCM


def make_labels():
    train_images = []
    train_labels = []

    for directory_path in glob.glob("train/*"):
    #for directory_path in glob.glob("train/mark/*.pgm"):
        # ['train', 'mark'], ['train', 'nomark']
        label = directory_path.split("/")[-1]
        print(label)
        for img_path in glob.glob(os.path.join(directory_path, "*.pgm")):
            print(img_path)
            
