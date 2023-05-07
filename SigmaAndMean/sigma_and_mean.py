#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def create_gaussian_image(shape:tuple, center:tuple, sigma:float):
    x, y = np.meshgrid(np.linspace(0, shape[1]-1, shape[1]),
                       np.linspace(0, shape[0]-1, shape[0]))
    d = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    g = np.exp(-( (d)**2 / (2.0 * sigma**2 ) ))
    g = (g * 255).astype(np.uint8)

    return g


def save_pgm_image(filename:str, array:np.ndarray):
    with open(filename, 'w') as f:
        f.write('P2\n')
        f.write(f"{array.shape[1]} {array.shape[0]}\n")
        f.write("255\n")
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                f.write(f"{array[i, j]} ")
                if j % 16 == 0:
                    f.write("\n")


def create_randomized_image(array:np.ndarray):
    flat_array = array.flatten()
    np.random.shuffle(flat_array)
    return flat_array.reshape(array.shape)


def plot_histograms(img1:np.ndarray,
                    img2:np.ndarray,
                    title1:str='Image 1',
                    title2:str='Image 2'):
    plt.subplot(121)
    plt.hist(img1.ravel(), bins=256, range=(0, 255), density=True, color='gray')
    plt.title(title1)
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')

    plt.subplot(122)
    plt.hist(img2.ravel(), bins=256, range=(0, 255), density=True, color='gray')
    plt.title(title2)
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')

    plt.show()


def main():
    shape = (512, 512)
    center = (256, 256)
    sigma = 128

    # 1つ目の画像を作成
    gaussian_image = create_gaussian_image(shape, center, sigma)
    save_pgm_image('gaussian_image.pgm', gaussian_image)
    gmean = gaussian_image.mean()
    gstdev = gaussian_image.std()

    # 2つ目の画像を作成
    randomized_image = create_randomized_image(gaussian_image)
    save_pgm_image('randomized_image.pgm', randomized_image)
    rmean = randomized_image.mean()
    rstdev = randomized_image.std()

    # 画像を表示
    plt.subplot(121)
    plt.imshow(gaussian_image, cmap='gray')
    plt.title(f'Gaussian Image\nmean: {gmean:.2f}, stdev: {gstdev:.2f}')

    plt.subplot(122)
    plt.imshow(randomized_image, cmap='gray')
    plt.title(f'Randomized Image\nmean: {rmean:.2f}, stdev: {rstdev:.2f}')

    plt.show()

    plot_histograms(gaussian_image, randomized_image,
                    'Gaussian Image', 'Randomized Image')


if __name__ == '__main__':
    main()
