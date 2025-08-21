"""
@author: LEI Manchun
"""
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
from single_image_based_vignetting_estimation import estimate_vignetting, correct_vignetting, compute_vignetting_model

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
gray_image = np.asarray(Image.open(os.path.join(current_dir, 'input.jpg')).convert('RGB').convert('L')).astype(float)
center, param, entropies = estimate_vignetting(gray_image, bins=256, max_value=255)
g = compute_vignetting_model(center, gray_image.shape, param)
x = 1/g
vignetting_image = 255 * (x - x.min())/(x.max() - x.min())

Image.fromarray(vignetting_image.astype(np.uint8), mode='L').save(os.path.join(current_dir, 'vignetting_model.jpg'))

correct_vignetting(os.path.join(current_dir, 'input.jpg'), os.path.join(current_dir, 'output.jpg'))