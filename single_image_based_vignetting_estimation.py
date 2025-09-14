"""
@author: LEI Manchun
"""
import math
import numpy as np
from PIL import Image
def compute_centroid(gray_image):
    rows, cols = gray_image.shape
    y_indices, x_indices = np.indices((rows, cols))
    total = gray_image.sum()
    cm_x = (x_indices * gray_image).sum() / total
    cm_y = (y_indices * gray_image).sum() / total
    return cm_x, cm_y

def compute_max_distance(center, dimension):
    cm_x, cm_y = center
    rows, cols = dimension
    corners = [
        (0, 0),          
        (0, rows),        
        (cols, 0),        
        (cols, rows)  
    ]
    distances = []
    for (x, y) in corners:
        dx = x - cm_x
        dy = y - cm_y
        dist = np.sqrt(dx*dx + dy*dy)
        distances.append(dist)
    return max(distances)

def compute_vignetting_model(center, dimension, parameter_tup):
    cm_x, cm_y = center
    rows, cols = dimension
    y, x = np.indices((rows, cols))
    distance = np.sqrt((x - cm_x) ** 2 + (y - cm_y) ** 2)
    max_distance = compute_max_distance(center, dimension)
    r = distance / max_distance
    a, b, c = parameter_tup
    g = 1 + a * r**2 + b * r**4 + c * r**6
    return g

def compute_histogram(gray_image, bins=256):
    intensity = np.log1p(np.clip(gray_image, 0, None))
    intensity = (intensity - intensity.min()) / (intensity.max() - intensity.min())
    hist, _ = np.histogram(intensity, bins=bins, range=(0,1))
    return hist.astype(np.float32)

def compute_entropy(gray_image, bins=256):
    hist = compute_histogram(gray_image, bins=bins)
    p = hist / np.sum(hist)       
    p = p[p > 0]                
    H = -np.sum(p * np.log(p)) 
    return H

def compute_overflow(gray_image, max_value=255):
    mask = (gray_image > max_value)
    ratio = np.sum(mask)/mask.size
    return ratio

def check_monotonically_increase(parameter_tup, eps=1e-12):
    """Check if a, b, c could let g(r) = 1 + a r^2 + b r^4 + c r^6 monotonically increase in [0,1]"""
    a, b, c = parameter_tup
    if abs(c) < eps:  # treat c==0
        return a >= 0 and a + 2*b >= 0 and not (abs(a) < eps and abs(b) < eps)
    discriminant = 4*b**2 - 12*a*c
    if c < 0:
        if b**2 > 3*a*c:
            sqrt_disc = math.sqrt(discriminant)
            q_plus = (-2*b + sqrt_disc) / (6*c)
            q_minus = (-2*b - sqrt_disc) / (6*c)
            return q_plus <= 0 and q_minus >= 1
        return False
    if c > 0:
        if b**2 < 3*a*c:
            return True
        elif abs(b**2 - 3*a*c) < eps:
            return b >= 0 or 3*c + b <= 0
        else:
            sqrt_disc = math.sqrt(discriminant)
            q_plus = (-2*b + sqrt_disc) / (6*c)
            q_minus = (-2*b - sqrt_disc) / (6*c)
            return q_plus <= 0 or q_minus >= 1

def pil_resize(arr:np.ndarray, size: tuple[int,int], method=Image.BILINEAR) -> np.ndarray:
    return np.array(Image.fromarray(arr).resize(size, method))

def estimate_vignetting(gray_image, bins=256, max_value=255):
    """
    Coordinate Descent Method
    Compute the vignetting center
    Find a, b, c that could minimize the entropy of the image
    """
    gray_image = gray_image.astype(np.float32)
    center = compute_centroid(gray_image)
    if gray_image.shape[0]>300:
        gray_image_sm = pil_resize(gray_image, (round(gray_image.shape[1] * 300 / gray_image.shape[0]), 300))
        center_sm = compute_centroid(gray_image_sm)
    else:
        gray_image_sm = gray_image
        center_sm = center
    a = b = c = 0
    delta = 2
    min_H = np.Inf
    tol_abs = 0
    tol_rel = 0
    explored = set()
    entropies = []
    while delta > 1 / 256:
        initial_tup = (a, b, c)
        for parameter_tup in [(a, b, c + delta), (a, b, c - delta),
                              (a, b + delta, c), (a, b - delta, c), 
                              (a + delta, b, c), (a - delta, b, c)]:
            if parameter_tup not in explored:
                explored.add(parameter_tup)
                if check_monotonically_increase(parameter_tup):
                    g = compute_vignetting_model(center_sm, gray_image_sm.shape, parameter_tup)
                    corrected_image = gray_image_sm * g
                    E = compute_entropy(corrected_image, bins=bins) 
                    O = compute_overflow(corrected_image, max_value)
                    curr_H = E + O
                    # if the entropy is lower than current minimum - tolerence, update parameters to current ones
                    delta_H = min_H - curr_H
                    if (delta_H > tol_abs) or (delta_H / max(min_H, 1e-12) > tol_rel):
                        min_H = curr_H 
                        a, b, c = parameter_tup
                        entropies.append(min_H)
        # reduce the delta
        if initial_tup == (a, b, c):
            delta /= 2

    return center, (a, b, c), entropies

def correct_vignetting(srcfile, dstfile):
    bins = 256
    max_value = 255
    img = Image.open(srcfile)
    img_gray = img.convert("L")
    arr_rgb = np.array(img)
    arr_gray = np.array(img_gray)
    center, param, _ = estimate_vignetting(arr_gray, bins=bins, max_value=max_value)
    g = compute_vignetting_model(center, arr_gray.shape, param)
    arr_corrected = np.empty(arr_rgb.shape,dtype=np.uint8)
    for i in range(3):
        y = arr_rgb[:,:,i].astype(np.float32) * g
        y = np.clip(y,0,max_value)
        arr_corrected[:,:,i] = y.astype(np.uint8)
        
    Image.fromarray(arr_corrected).save(dstfile)

