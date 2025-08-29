'''
This file serves as a dumping ground for all standard video metric calculations.
'''
import sys
import os
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from skimage.metrics import mean_squared_error
class VideoMetrics:
    def __init__(self):
        pass

    def psnr(self, im, gt_im, data_range=None) -> float:
        return peak_signal_noise_ratio(im, gt_im, data_range)

    def ssim(self, im, gt_im, data_range=None) -> float:
        return structural_similarity(im, gt_im, data_range)

    def mse(self, im, gt_im) -> float:
        return mean_squared_error(im, gt_im)
    
    def vmaf(self, im) -> float:
        pass