#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import traceback

import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import block_reduce, shannon_entropy
from PIL import PpmImagePlugin

from sklearn import preprocessing


class GLCM():

    def __init__(self) -> None:
        self.__FIN = None
        self.__LABEL = None
        self.__PATCH_SIZE = 35
        self.__LEVELS = 4096
        self.__DISTANCES = [1, 3, 5]
        self.__ANGLES = [0, np.pi/4, np.pi/2, 3*np.pi/4]

    @property
    def fin(self):
        return self.__FIN

    @fin.setter
    def fin(self, fin:str):
        self.__FIN = fin

    @property
    def label(self):
        return self.__LABEL

    @label.setter
    def label(self, label:int):
        self.__LABEL = label

    @property
    def PATCH_SIZE(self):
        return self.__PATCH_SIZE

    @PATCH_SIZE.setter
    def PATCH_SIZE(self, patch_size):
        self.__PATCH_SIZE = patch_size

    @property
    def levels(self):
        return self.__LEVELS

    @levels.setter
    def levels(self, new_level:int):
        self.__LEVELS = new_level

    @property
    def DISTANCES(self):
        return self.__DISTANCES

    @DISTANCES.setter
    def DISTANCES(self, distances):
        self.__DISTANCES = distances

    @property
    def ANGLES(self):
        return self.__ANGLES

    @ANGLES.setter
    def ANGLES(self, angles):
        self.__ANGLES = angles


    def read_pgm_12bit(self, fin:str) -> None:
        # PGMヘッダー情報を読み込む
        self.fin = fin
        with open(fin, 'rb') as f:
            header = f.readline().decode().strip()
            if header != 'P2':
                raise ValueError("Not a valid PGM file")

            while True:
                size_line = f.readline().decode().strip()
                if size_line.startswith('#'):
                    continue
                else:
                    break

            width, height = map(int, size_line.split())
            max_value = int(f.readline().decode().strip())

            if max_value != 4095:
                raise ValueError("Not a 12bit level grayscale PGM image")

        # scikit-imageのio.imread()で画像データを読み込む
        self.img = io.imread(fin, 
                             as_gray=True,
                             plugin='pil',
                             dtype=np.uint16)
        # 読み込んだデータはuint16形式のデータとなるので、
        # 輝度が16ビットになっている。
        # 12ビットの輝度を保つ為に、4ビット分だけ右ビットシフトする
        self.img = self.img >> 4


    def imshow(self, vmin=0, vmax=5, save_img=False, logplot=True) -> None:
        fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(12,4))
        vmin = np.log(self.GLCM.min()) if self.GLCM.min() > 0    else 0
        vmax = np.log(self.GLCM.max()) if self.GLCM.max() < 4095 else np.log(4095)

        glcm = self.GLCM
        if logplot:
            # 0以下の値を持つインデックスを取得。基本的には0が最小値のはずではある
            zero_indices = glcm <= 0
            # log計算のために0以下の値を小さな正の値に置き換える
            glcm[zero_indices] = 1
            # logを適用する
            glcm = np.log10(glcm)
            # 元の0以下の値を持つ要素を0に置きかえる
            glcm[zero_indices] = 0

        axs[0].imshow(self.img, cmap='gray')
        axs[1].imshow(glcm[:, :, 0, 0], vmin=vmin, vmax=vmax)
        axs[2].imshow(glcm[:, :, 0, 1], vmin=vmin, vmax=vmax)
        axs[3].imshow(glcm[:, :, 0, 2], vmin=vmin, vmax=vmax)
        im = axs[4].imshow(glcm[:, :, 0, 3], vmin=vmin, vmax=vmax)
        name = os.path.splitext(self.fin)[0]
        plt.suptitle(name + f"\n vmin:{vmin} vmax:{vmax}")
        plt.tight_layout()

        if save_img:
            savename = name + "_glcm.png"
            plt.savefig(savename)
        else:
            plt.show()
        plt.close()


    def calc_glcm(self, pooled=False) -> str:

        def _calc_entropy(glcm:object) -> float:
            entropies = []
            for i, distance  in enumerate(self.DISTANCES):
                entropy = 0
                for j, angle in enumerate(self.ANGLES):
                    entropy += shannon_entropy(self.GLCM[:, :, i, j])

                # 各Distanceは角度別データの平均値とする
                entropies.append(entropy / len(self.DISTANCES))

            return entropies

        if pooled:
            img = self.pooled_img
        else:
            img = self.img

        self.GLCM = graycomatrix(img,
                                 self.DISTANCES,
                                 self.ANGLES,
                                 levels=self.levels)

        disssimilarity = graycoprops(self.GLCM, 'dissimilarity').mean(axis=1)
        correlation    = graycoprops(self.GLCM, 'correlation').mean(axis=1)
        energy         = graycoprops(self.GLCM, 'energy').mean(axis=1)
        homogeneity    = graycoprops(self.GLCM, 'homogeneity').mean(axis=1)
        contrast       = graycoprops(self.GLCM, 'contrast').mean(axis=1)
        entropy        = _calc_entropy(self.GLCM)

        DSS = [str(disssimilarity[i]) for i in range(len(disssimilarity))]
        COR = [str(correlation[i])    for i in range(len(correlation))]
        ENG = [str(energy[i])         for i in range(len(energy))]
        HMG = [str(homogeneity[i])    for i in range(len(homogeneity))]
        CTR = [str(contrast[i])       for i in range(len(contrast))]
        ENT = [str(entropy[i])        for i in range(len(entropy))]
        
        result = f"{self.fin},{self.label}," + ",".join(DSS+COR+ENG+HMG+CTR+ENT) + "\n"
        #print(result)
        #print(self.fin, self.label,
        #      ",".join(DSS+COR+ENG+HMG+CTR+ENT), sep=",")
        return result


    def apply_pooling(self, 
                method:str='average', 
                new_levels:int=256, 
                padding_size:int=0,
                padding_mode:str='edge') -> None:

        def _pad_image(img:np.array,
                       size:int = 1,
                       mode:str='edge') -> np.array:

            padded_img = np.pad(img, pad_width=size, mode=mode)
            return padded_img

        if padding_size > 0:
            img = _pad_image(self.img, size=padding_size, mode=padding_mode)
        else:
            img = self.img

        # paddingした範囲は直接の処理対象から外すことで、pooling後の
        # 画像サイズを所望のものとすることが出来る
        if method == 'max':
            self.pooled_img = block_reduce(img[padding_size:-padding_size,
                                               padding_size:-padding_size]\
                                               if padding_size > 0 else img,
                                           (2, 2), np.max)
        elif method == 'min':
            self.pooled_img = block_reduce(img[padding_size:-padding_size,
                                               padding_size:-padding_size]\
                                               if padding_size > 0 else img,
                                           (2, 2), np.min)
        elif method == 'average':
            self.pooled_img = block_reduce(img[padding_size:-padding_size,
                                               padding_size:-padding_size]\
                                               if padding_size > 0 else img,
                                           (2, 2), np.mean)
        elif method == 'median':
            self.pooled_img = block_reduce(img[padding_size:-padding_size,
                                               padding_size:-padding_size]\
                                               if padding_size > 0 else img,
                                           (2, 2), np.median)
        else:
            raise ValueError("Invalid pooling method. Choose from 'max', 'min', 'average', or 'median'.")

        # 諧調変換
        max_value = np.max(self.pooled_img)
        new_max_value = new_levels - 1
        self.pooled_img = (self.pooled_img / max_value) * new_max_value
        self.pooled_img = self.pooled_img.astype(np.uint16)
        self.levels = new_levels


    def run(self, 
            fin:str, 
            pooling=False,
            padding_size=0,
            save_img=False,
            vmin=0, vmax=100) -> str:

        self.read_pgm_12bit(fin)
        if pooling:
            self.apply_pooling(padding_size=padding_size)

        result = self.calc_glcm(pooled=pooling)

        if save_img:
            self.imshow(vmin=vmin, vmax=vmax, save_img=save_img)

        return result


    def run_all(self,
                pooling=False,
                padding_size=0, save_img=False,
                vmin:float=0, vmax:float=100) -> None:

        try:                             
            with open("result.txt", "w") as f:
                for directory_path in glob.glob("train/*"):
                    label = directory_path.split("/")[-1]
                    self.label = label
                    for img_path in glob.glob(os.path.join(directory_path, "*.pgm")):
                        print(img_path)
                        result = self.run(img_path,
                                          pooling=pooling,
                                          padding_size=padding_size,
                                          save_img=save_img,
                                          vmin=vmin, vmax=vmax)
                        f.write(result)

        except Exception as e:
            print(e)
            print(traceback.format_exc())


if __name__ == '__main__':
    glcm = GLCM()
    glcm.run_all(
        pooling=True,
        padding_size=0,
        save_img=False, 
    )
