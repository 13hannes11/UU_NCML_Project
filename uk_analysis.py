#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#This code is modified to run in Kaggle

import voting_lib.load_data as ld
import voting_lib.voting_analysis as va
import numpy as np
import pandas as pd
import os



# Train model
grid_h = 30       # Grid height
grid_w = 30       # Grid width
radius = 3        # Neighbour radius
step = 0.5
ep = 100         # No of epochs


main_directory = 'uk/csv'
for dirname, _, filenames in os.walk(main_directory):
        if dirname == main_directory: #to skip main directory path 
            continue
        else:
            # Load data
            data = ld.load_uk_data(dirname).to_numpy() 

            X = data[:,2:]

            model = va.train_model(X, grid_h, grid_w, radius, step, ep)
            # Predict and visualize output
            va.predict(model, data, grid_h, grid_w)