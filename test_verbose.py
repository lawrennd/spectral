#!/usr/bin/env python3
"""Test verbose output on ear dataset."""

import time
import numpy as np
from PIL import Image
from spectral import SpectralCluster

# Load ear data  
print("Loading ear dataset...")
img = Image.open('examples/data/ear.bmp')
img_array = np.array(img)
if len(img_array.shape) == 3:
    img_array = img_array[:, :, 0]
y_coords, x_coords = np.where(img_array < 128)
X = np.column_stack([x_coords, y_coords])

print(f"Dataset size: {X.shape[0]} points\n")

# Run with verbose output
clf = SpectralCluster(sigma=0.707, random_state=1, verbose=True)
t0 = time.time()
clf.fit(X)
print(f'\nTotal time: {time.time()-t0:.1f}s')
print(f'Found {clf.n_clusters_} clusters')
