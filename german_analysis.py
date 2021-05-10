#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import load_data as ld
import voting_analysis as va

# Load data
data = ld.load_german_data().to_numpy()
X = data[:,2:]

# Train model
grid_h = 10       # Grid height
grid_w = 10       # Grid width
radius = 2        # Neighbour radius
step = 0.5
ep = 300          # No of epochs

model = va.train_model(X, grid_h, grid_w, radius, step, ep)

# Predict and visualize output
va.predict(model, data, grid_h, grid_w)