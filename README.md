# single-image-vignetting-correction
This repository provides a **single-image vignetting correction** tool.

## Background
The correction is based on the **minimum log entropy optimization** approach from prior research.  

## My Contribution
I introduce a **overflow penalty metric** that measures the proportion of pixels exceeding the maximum intensity after correction.  
This penalty helps prevent overly steep vignetting models and avoids saturation in practical applications.

## Main program
- `single_image_based_vignetting_estimation.py`  
  Estimates the vignetting model parameters:
  - center (cm_x, cm_y)
  - polynomial coefficients (a, b, c)

## Workflow
1. Estimate vignetting parameters from a single image.
2. Apply correction to obtain a vignetting-free image.

## Features
- Single-image based (no calibration dataset required)
- Minimum log entropy optimization
- Overflow penalty metric
- Polynomial vignetting model: `1 + a*r^2 + b*r^4 + c*r^6`
