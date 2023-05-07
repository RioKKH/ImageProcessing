#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import traceback
from pathlib import Path

import numpy as np
from skimage import io
from skimage import exposure
from skimage.util import random_noise


class DataAugmentor:

    def __init__(self) -> None:
        eps = 1.0E-6
        self.fin = None
        #self.input_dir = input_dir
        #self.output_dir = output_dir
        self.offset_values = [-500, 500]
        self.gain_values   = [0.7, 1.3]
        self.uniform_noise_intensities  = [1000, 2000, 3000]
        self.gaussian_noise_intensities = [10000, 20000, 30000]
        self.linear_strengths = [1500,]
        #self.linear_strengths = [500, 1000, 1500]
        self.linear_directions = [0, 45, 90, 135, 180, 225, 270, 315]
        self.quadratic_strengths = [1000, 1500, 2000]


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


    def save_pgm_p2(self, filename:str, img:np.array):
        print(filename)
        max_value = 2**12-1 # 12-bitの最大値
        height, width = img.shape

        # P2形式のヘッダーを書き込む
        with open(filename, "w") as f:
            f.write("P2\n")
            f.write(f"{width} {height}\n")
            f.write(f"{max_value}\n")

            # 各ピクセルの値を書き込む
            for i in range(height):
                for j in range(width):
                    f.write(f"{img[i, j]} ")
                    if j % 16 == 0:
                        f.write("\n")


    def check_file_existence(self, directory, target_file):
        path = Path(directory)
        for filename in path.iterdir():
            if filename.name == target_file:
                return True
        return False


    def adjust_brightness(self, gain=1, offset=0):
        adjusted_image = self.img * gain + offset
        return np.clip(adjusted_image, 0, 4095)


    def add_uniform_noise(self, intensity):
        noise = np.random.uniform(-intensity, intensity, size=self.img.shape)
        noisy_image = self.img + noise
        return np.clip(noisy_image, 0, 4095)


    def add_gaussian_noise(self, intensity=1, mean=0, var=0.01):
        noise = random_noise(self.img, mode='gaussian',
                             mean=mean, var=var, clip=True)
        noisy_image = self.img + noise * intensity
        return np.clip(noisy_image, 0, 4095)


    def adjust_linear_gradient(self, strength=0, direction=0):
        if direction == 0: # 0-degree. right side of image is brighter.
            x = np.linspace(-strength, strength, self.img.shape[1])
            y = np.full((1, self.img.shape[0]), 0)
        elif direction == 180: # 180-degree. left side of image is brighter.
            x = np.linspace(strength, -strength, self.img.shape[1])
            y = np.full((1, self.img.shape[0]), 0)
        elif direction == 90: # 90-degree. upper side of image is brighter.
            x = np.full((1, self.img.shape[1]), 0)
            y = np.linspace(strength, -strength, self.img.shape[0])
        elif direction == 270: # 270-degree. lower side of image is brighter.
            x = np.full((1, self.img.shape[1]), 0)
            y = np.linspace(-strength, strength, self.img.shape[0])
        elif direction == 45: # 45-degree. upper-right side of image is brighter.
            x = np.linspace(-strength, strength, self.img.shape[1])
            y = np.linspace(strength, -strength, self.img.shape[0])
        elif direction == 225: # 225-degree. lower-left side of image is brighter.
            x = np.linspace(strength, -strength, self.img.shape[1])
            y = np.linspace(-strength, strength, self.img.shape[0])
        elif direction == 135: # 135-degree. upper-left side of image is brighter.
            x = np.linspace(strength, -strength, self.img.shape[1])
            y = np.linspace(strength, -strength, self.img.shape[0])
        elif direction == 315: # 315-degree. lower-right side of image is brighter.
            x = np.linspace(-strength, strength, self.img.shape[1])
            y = np.linspace(-strength, strength, self.img.shape[0])
        else:
            raise ValueError("Invalid direction. Must be one of intergers from 1 to 4.")

        xv, yv = np.meshgrid(x, y)
        gradient = xv + yv
        return np.clip(self.img + gradient, 0, 4095)


    def adjust_quadratic_gradient(self, strength):
        x = np.linspace(-strength, strength, self.img.shape[1])
        y = np.linspace(-strength, strength, self.img.shape[0])
        xv, yv = np.meshgrid(x, y)
        gradient = -np.sqrt(xv**2 + yv**2)
        return np.clip(self.img + gradient, 0, 4095)


    def augment(self, fin:str) -> None:

        self.read_pgm_12bit(fin)

        path = Path(fin)
        dirname = path.parent
        filename = path.stem
        print(f"{dirname}, {filename}")

        for gain in self.gain_values:
            for offset in self.offset_values:
                target = f"{filename}_gain_{gain}_offset_{offset}.pgm"
                if self.check_file_existence(dirname, target):
                    continue
                else:
                    adjusted = self.adjust_brightness(gain=gain, offset=offset)
                    self.save_pgm_p2(
                        os.path.join(dirname, target), adjusted.astype(np.uint16)
                    )

        for noise in self.uniform_noise_intensities:
            target = f"{filename}_uniform_noise_{noise}.pgm"
            if self.check_file_existence(dirname, target):
                continue
            else:
                noisy_img = self.add_uniform_noise(noise)
                self.save_pgm_p2(
                    os.path.join(dirname, target), noisy_img.astype(np.uint16)
                )

        #for noise in self.gaussian_noise_intensities:
        #    noisy_img = self.add_gaussian_noise(noise)
        #    self.save_pgm_p2(
        #        os.path.join(dirname, f"{filename}_gaussian_noise_{noise}.pgm"),
        #        noisy_img.astype(np.uint16)
        #    )

        for direction in self.linear_directions:
            for strength in self.linear_strengths:
                target = f"{filename}_linear_gradient_{direction}_{strength}.pgm"
                if self.check_file_existence(dirname, target):
                    continue
                else:
                    adjusted = self.adjust_linear_gradient(strength=strength,
                                                           direction=direction)
                    self.save_pgm_p2(
                        os.path.join(dirname, target), adjusted.astype(np.uint16)
                    )

        #for strength in self.quadratic_strengths:
        #    adjusted = self.adjust_quadratic_gradient(strength)
        #    self.save_pgm_p2(
        #        os.path.join(dirname, f"{filename}_quadratic_gradient_{strength}.pgm"),
        #        adjusted.astype(np.uint16)
        #    )

    def run(self):
        try:
            img_path = "train/mark/mensrch_-1_-1_-10.pgm"
            self.augment(img_path)

        except Exception as e:
            print(e)
            print(traceback.format_exc())


    def run_all(self):
        try:
            for directory_path in glob.glob("train/*"):
                label = directory_path.split("/")[-1]
                self.label = label
                for img_path in glob.glob(os.path.join(directory_path, "*.pgm")):
                    self.augment(img_path)

        except Exception as e:
            print(e)
            print(traceback.format_exc())


