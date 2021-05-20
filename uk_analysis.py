#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#This code is modified to run in Kaggle

import voting_lib.load_data as ld
import voting_lib.voting_analysis as va
import voting_lib.political_compass as pc
import numpy as np
import pandas as pd
import os



# Train model
grid_h = 30       # Grid height
grid_w = 30       # Grid width
radius = 3        # Neighbour radius
step = 0.5
ep = 1         # No of epochs


period_to_compass_year = {'2015_uk':2015, '2017_uk':2017, '2019_uk':2019}
main_directory = 'uk/csv'
for dirname, _, filenames in os.walk(main_directory):
        if dirname == main_directory: #to skip main directory path 
            continue
        elif os.path.isdir(dirname):
            # Load data
            data = ld.load_uk_data(dirname).to_numpy() 

            X = data[:,2:]

            model = va.train_model(X, grid_h, grid_w, radius, step, ep)
            # Predict and visualize output
            va.predict(model, data, grid_h, grid_w, pc.get_compass_parties(year=period_to_compass_year[dirname.split('/')[-1]], country='uk'))