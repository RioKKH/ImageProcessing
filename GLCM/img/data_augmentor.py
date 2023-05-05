#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import numpy as np
from skimage import io
from skimage import exposure
from skimage.util import random_noise


class DataAugmentor:

    def __init__(self, input_dir, output_dir):
        self.fin = None
        self.input_dir = input_dir
        self.output_dir = output_dir

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)


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


    def adjust_offset(self, image, offset):
        return exposure.adjust_gamma(image, 1, offset)


    def adjust_gain(self, image, gain):
        return exposure.adjust_gamma(image, gain)


    def add_uniform_noise(self, image, intensity):
        return random_noise(image, mode='uniform', clip=True, var=intensity)


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

    def augment(self, offset_values, gain_values, uniform_noise_values, linear_gradient_strengths, quadratic_gradient_strengths):
        for filename in os.listdir(self.input_dir):
            if filename.endswith('.pgm'):
                img = io.imread(os.path.join(self.input_dir, filename), as_gray=True)

                for offset in offset_values:
                    adjusted = self.adjust_offset(img, offset)
                    io.imsave(os.path.join(self.output_dir, f"{filename[:-4]}_offset_{offset}.pgm"), adjusted)

                for gain in gain_values:
                    adjusted = self.adjust_gain(img, gain)
                    io.imsave(os.path.join(self.output_dir, f"{filename[:-4]}_gain_{gain}.pgm"), adjusted)

                for noise in uniform_noise_values:
                    noisy_img = self.add_uniform_noise(img, noise)
                    io.imsave(os.path.join(self.output_dir, f"{filename[:-4]}_uniform_noise_{noise}.pgm"), noisy_img)

                for strength in linear_gradient_strengths:
                    adjusted = self.adjust_linear_gradient(img, strength)
                    io.imsave(os.path.join(self.output_dir, f"{filename[:-4]}_linear_gradient_{strength}.pgm"), adjusted)

                for strength in quadratic_gradient_strengths:
                    adjusted = self.adjust_quadratic_gradient(img, strength)
                    io.imsave(os.path.join(self.output_dir, f"{filename[:-4]}_quadratic_gradient_{strength}.pgm"), adjusted)
