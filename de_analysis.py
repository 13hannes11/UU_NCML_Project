#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import voting_lib.load_data as ld
import voting_lib.voting_analysis as va
import voting_lib.political_compass as pc
from voting_lib.party_colors import de_name_color
import numpy as np

# Training Paramters
# Grid size is chosen such that node count = 5*sqrt(N)
grid_h = 11       # Grid height
grid_w = 11       # Grid width
radius = 2        # Neighbour radius
step = 0.5        # Learning step
ep = 500          # No of epochs

# Load data
dataset = ld.load_german_data()

period_to_compass_year = {17:2009, 18:2013, 19:2017}

for period, df in dataset.items():

    print("Election Period ", period)
    data = df.to_numpy()
    
    X = data[:,2:]

    # Train model
    model = va.train_model(X, grid_h, grid_w, radius, step, ep)

    # Predict and visualize output
    va.predict(model, data, grid_h, grid_w, de_name_color, pc.get_compass_parties(year=period_to_compass_year[period], country='de'))