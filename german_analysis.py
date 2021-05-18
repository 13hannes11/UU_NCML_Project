#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import voting_lib.load_data as ld
import voting_lib.voting_analysis as va
import numpy as np

# Training Paramters
# Grid size is chosen such that node count = 5*sqrt(N)
grid_h = 11       # Grid height
grid_w = 11       # Grid width
radius = 2        # Neighbour radius
step = 0.5
ep = 300          # No of epochs

# Load data
dataset = ld.load_german_data()

for period, df in dataset.items():

    print("Election Period ", period)
    data = df.to_numpy()
    
    X = data[:,2:]

    # Train model
    model = va.train_model(X, grid_h, grid_w, radius, step, ep)

    # Predict and visualize output
    va.predict(model, data, grid_h, grid_w)