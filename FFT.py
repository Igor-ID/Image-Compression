from matplotlib.image import imread
import numpy as np
import matplotlib.pyplot as plt
import os

"""Image compression using FFT"""

plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams.update({'font.size': 18})
im = imread('data/dog.jpg')
im_gray = np.mean(im, -1)  # convert RGB to gray scale
# img = plt.imshow(im_gray)
# img.set_cmap('gray')
# plt.axis('off')
# Apply FFT2 to the gray scale (1 dimensional) image
imfft = np.fft.fft2(im_gray)
# Sort the FFT2 values by magnitude, taking the absolute value
# (get rid of the imaginary part geometrically => abs(3 +- 2j) == sqrt(9 + 4))
imfftsort = np.sort(np.abs(imfft.reshape(-1)))
# Descending order
# imffsort_rev = imfftsort[::-1]
# Zero out all small coefficients and inverse transform
# I'm going to keep first 10%, 5%, 1% and 0.2% of Fourier coefficients
for keep in (0.1, 0.05, 0.01, 0.002):
    thresh_ind = int(np.floor((1-keep)*len(imfftsort)))
    thresh = imfftsort[thresh_ind]
    # Find True(1) or False(0) in vector of Fourier coefficients
    ind = np.abs(imfft) > thresh
    # Get rid of all Fourier coefficients, except first keep(10, 5, 1 and 0.2 percents respectively)
    im_gray_low = imfft * ind
    # apply inverse FFT to get processed image
    im_low = np.fft.ifft2(im_gray_low).real
    plt.figure()
    plt.imshow(im_low, cmap='gray')
    plt.axis('off')
    plt.title('Compressed image: keep = ' + str(keep*100) + '%')


plt.show()




