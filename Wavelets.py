import pywt
import matplotlib.pyplot as plt
from matplotlib.image import imread
import numpy as np

"""Discrete Wavelet transform"""

plt.rcParams['figure.figsize'] = [8, 8]
plt.rcParams.update({'font.size': 18})

im = imread('data/dog.jpg')
im_gray = np.mean(im, -1)  # convert RGB to gray scale

# Wavelet decomposition (2 level)
# 2 level wavelet
n = 2
# show all discrete wavelets in pywavelet module:
print(pywt.wavelist(kind='discrete'))
# Use Daubechies 1 wavelet family.
# It looks like all first level types of discrete wavelets show best result in subplot landscape
w = 'db1'
coeffs = pywt.wavedec2(im_gray, wavelet=w, level=n)

# normalize each coefficient array
coeffs[0] /= np.abs(coeffs[0]).max()
for detail_level in range(n):
    coeffs[detail_level + 1] = [d/np.abs(d).max() for d in coeffs[detail_level + 1]]

arr, coeff_slices = pywt.coeffs_to_array(coeffs)
plt.imshow(arr, cmap='gray', vmin=-0.25, vmax=0.75)
plt.show()














