#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import traceback

import numpy as np
from skimage import io
from skimage import exposure
from skimage.util import random_noise


class DataAugmentor:

    def __init__(self):
        eps = 1
        self.fin = None
        #self.input_dir = input_dir
        #self.output_dir = output_dir
        self.offset_values = np.arange(-200, 200+eps, 100)
        #self.gain_values = np.arange(-


    def read_pgm_12bit(self, fin:str) -> None:
        self.fin = fin
        with open(fin, 'rb') as f:
            header = f.readline().decode().strip()
            if header != 'P2':
                raise ValueError("Not a valid PGM file")

            while True:
                size_line = f.readline().decode().strip()
                if size_line.startswith('#'):
                    continue # コメント行はスキップする
                else:
                    break # コメント行を全部スキップしたのでwhileループから抜ける

            width, height = map(int, size_line.split())
            max_value = int(f.readline().decode().strip())

            if max_value != 4095:
                raise ValueError("Not a 12bit level grayscale PGM image")

        # scikit-imageのio.imread()で画像データを読み込む
        self.img = io.imread(fin, as_gray=True, plugin='pil', dtype=np.uint16)
        self.img = self.img >> 4


    def adjust_brightness(self, gain, offset):
        adjusted_image = self.img * gain + offset
        return np.clip(adjusted_image, 0, 4095)


    def add_uniform_noise(self, intensity):
        noise = np.random.uniform(-intensity, intensity, size=self.img.shape)
        noise_image = self.img + noise
        return np.clip(noise_image, 0, 4095)


    def add_gaussian_noise(self, mean=0, var=0.01):
        return random_noise(self.img,
                            mode='gaussian',
                            mean=mean, var=var, clip=True)


    def adjust_linear_gradient(self, image, strength):
        x = np.linspace(-strength, strength, image.shape[1])
        y = np.linspace(-strength, strength, image.shape[0])
        xv, yv = np.meshgrid(x, y)
        gradient = xv + yv
        return np.clip(image + gradient, 0, 1)


    def adjust_quadratic_gradient(self, image, strength):
        x = np.linspace(-strength, strength, image.shape[1])
        y = np.linspace(-strength, strength, image.shape[0])
        xv, yv = np.meshgrid(x, y)
        gradient = xv**2 + yv**2
        return np.clip(image + gradient, 0, 1)


    def augment(self, 
                fin:str,
                offset_values:float,
                gain_values:float,
                uniform_noise_values:float,
                linear_gradient_strengths:float,
                quadratic_gradient_strengths:float) -> None:

        self.read_pgm_12bit(fin)

        name = os.path.splitext(fin)[0]

        for offset in offset_values:
            adjusted = self.adjust_brightness(gain=1, offset=offset)
            io.imsave(
                os.path.join(self.output_dir, f"{name}_offset_{offset}.pgm"),
                adjusted
            )

        for gain in gain_values:
            adjusted = self.adjust_gain(gain=gain, offset=0.0)
            io.imsave(
                os.path.join(self.output_dir, f"{name}_gain_{gain}.pgm"),
                adjusted
            )

        for noise in uniform_noise_values:
            noisy_img = self.add_uniform_noise(noise)
            io.imsave(
                os.path.join(self.output_dir, f"{name}_uniform_noise_{noise}.pgm"),
                noisy_img
            )

        #for strength in linear_gradient_strengths:
        #    adjusted = self.adjust_linear_gradient(self.img, strength)
        #    io.imsave(
        #        os.path.join(self.output_dir, f"{name}_linear_gradient_{strength}.pgm"),
        #        adjusted
        #    )

        #for strength in quadratic_gradient_strengths:
        #    adjusted = self.adjust_quadratic_gradient(self.img, strength)
        #    io.imsave(
        #        os.path.join(self.output_dir, f"{name}_quadratic_gradient_{strength}.pgm"),
        #        adjusted
        #    )

    def run(self):
        try:
            img_path = "train/mark/mensrch_-1_-1_-10.pgm"
            self.augment(img_path)

        except Exception as e:
            print(e)


    def run_all(self):
        try:
            for directory_path in glob.glob("train/*"):
                label = directory_path.split("/")[-1]
                self.label = label
                for img_path in glob.glob(os.path.join(directory_path, "*.pgm")):
                    self.augment(img_path)

        except Exception as e:
            print(e)


