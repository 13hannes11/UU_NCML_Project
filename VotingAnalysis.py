#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from itertools import product
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from neupy import algorithms

# Loading German Parliament Votes
def load_german_data():
    title_file = "filename_to_titles.csv"
    vote_counter = -1
    data = pd.DataFrame()
    
    name_column = 'Bezeichnung'
    
    vote_column_to_title = {}
    
    voting_features = ['ja', 'nein', 'Enthaltung', 'ungültig']
    for dirname, _, filenames in os.walk('./de/csv'):
        for filename in filenames:
            if filename != title_file:
                vote_counter += 1
                df = pd.read_csv(os.path.join(dirname, filename))
                
                # Give each voting behaviour type an identifier from 0 to len(voting_features) - 1
                for i, feature in enumerate(voting_features):
                    df[feature] *= i
                vote_column_name = f'vote_{vote_counter}'
                # Map column name of vote to filename -> allows retrieving what the vote was about
                vote_column_to_title[vote_column_name] = filename
                
                # add feature for the vote
                df[vote_column_name] = df[voting_features].sum(axis=1)
                
                if data.empty:
                    # if first file that is loaded set data equal to data from first file
                    data = df[[name_column, vote_column_name]]
                else:
                    # merge data with already loaded data 
                    data = data.merge(df[[name_column, vote_column_name]], on=name_column)
    
    print(vote_column_to_title)
    print(data)
    return data

# Loading UK Parliament Votes
def load_uk_data():
    # Preprocess data
    vote_counter = -1
    data = pd.DataFrame()
    
    name_column = 'Member'
    vote_column = 'Vote'
    
    column_to_filename = {}
    
    voting_features = {'Aye':0, 'Teller - Ayes':0, 'No':1, 'Teller - Noes':1, 'No Vote Recorded':2}
    for dirname, _, filenames in os.walk('./uk/csv'):
        for filename in filenames:
            vote_counter += 1
            print(os.path.join(dirname, filename))
            
            # Read title rows
            title_df = pd.read_csv(os.path.join(dirname, filename),nrows=(3),skip_blank_lines=True,header=None)
        
            # Read data rows
            df = pd.read_csv(os.path.join(dirname, filename),skiprows=(10))
            
            # Give each voting behaviour type an identifier from 0 to len(voting_features) - 1
            df[vote_column].replace(voting_features, inplace=True)
            
            #Replace the vote column name
            vote_column_name = f'vote_{vote_counter}'
            df=df.rename(columns={vote_column:vote_column_name})
             
            # Map column name of vote to title -> allows retrieving what the vote was about
            column_to_filename[vote_column_name] = title_df.iat[2,0]
                    
            if data.empty:
                # if first file that is loaded set data equal to data from first file
                data = df[[name_column, vote_column_name]]
            else:
                # merge data with already loaded data 
                data = data.merge(df[[name_column, vote_column_name]], on=name_column)
    
    print(column_to_filename)
    print(data)
    return data

# Heatmap
def iter_neighbours(weights, hexagon=False):
    _, grid_height, grid_width = weights.shape

    hexagon_even_actions = ((-1, 0), (0, -1), (1, 0), (0, 1), (1, 1), (-1, 1))
    hexagon_odd_actions = ((-1, 0), (0, -1), (1, 0), (0, 1), (-1, -1), (1, -1))
    rectangle_actions = ((-1, 0), (0, -1), (1, 0), (0, 1))

    for neuron_x, neuron_y in product(range(grid_height), range(grid_width)):
        neighbours = []

        if hexagon and neuron_x % 2 == 1:
            actions = hexagon_even_actions
        elif hexagon:
            actions = hexagon_odd_actions
        else:
            actions = rectangle_actions

        for shift_x, shift_y in actions:
            neigbour_x = neuron_x + shift_x
            neigbour_y = neuron_y + shift_y

            if 0 <= neigbour_x < grid_height and 0 <= neigbour_y < grid_width:
                neighbours.append((neigbour_x, neigbour_y))

        yield (neuron_x, neuron_y), neighbours

def compute_heatmap(weight, grid_height, grid_width):
    heatmap = np.zeros((grid_height, grid_width))
    for (neuron_x, neuron_y), neighbours in iter_neighbours(weight):
        total_distance = 0

        for (neigbour_x, neigbour_y) in neighbours:
            neuron_vec = weight[:, neuron_x, neuron_y]
            neigbour_vec = weight[:, neigbour_x, neigbour_y]

            distance = np.linalg.norm(neuron_vec - neigbour_vec)
            total_distance += distance

        avg_distance = total_distance / len(neighbours)
        heatmap[neuron_x, neuron_y] = avg_distance

    return heatmap

def plot_mps(fig, ax, x, y, labels, colors, cmap = plt.cm.RdYlGn):
    ANNOTATION_DISTANCE = 5
    TRANSPARENCY = 0.8
    scatterplot = plt.scatter(x,y,c=colors, s=5, cmap=cmap)

    annot = ax.annotate("", xy=(0,0), 
                        xytext=(ANNOTATION_DISTANCE, ANNOTATION_DISTANCE),
                        textcoords="offset points",
                        bbox=dict(boxstyle="Square"))
    annot.set_visible(False)

    def update_annot(ind):
        index = ind["ind"][0]
        pos = scatterplot.get_offsets()[index]
        annot.xy = pos
        text = f'{labels[index]}'
        annot.set_text(text)
        annot.get_bbox_patch().set_facecolor(cmap(colors[index]))
        annot.get_bbox_patch().set_alpha(TRANSPARENCY)

    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = scatterplot.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)
    #plt.show()

#Simple SOFM for German
plt.style.use('ggplot')

# Load data
data = load_german_data().to_numpy()
X = data[:,1:]
print(X)

inp = X.shape[1]   # No of features (bill count)
h = 150            # Grid height
w = 150            # Grid width
rad = 2            # Neighbour radius
ep = 300           # No of epochs

# Create SOFM
sofmnet = algorithms.SOFM(
    n_inputs=inp,
    step=0.5,
    show_epoch=100,
    shuffle_data=True,
    verbose=True,
    learning_radius=rad,
    features_grid=(h,w),
)

sofmnet.train(X, epochs=ep)

#Visualizing Output
fig,ax = plt.subplots()

weight = sofmnet.weight.reshape((sofmnet.n_inputs, h, w))
heatmap = compute_heatmap(weight, h, w)
plt.imshow(heatmap, cmap='Greys_r', interpolation='nearest')
#plt.axis('off')
plt.colorbar()

prediction = sofmnet.predict(X)
print(f'prediction: {prediction}')
# converting to x and y coordinates
# TODO: verify that actually correct
xs = np.argmax(np.argmax(prediction.reshape(X.shape[0], h, w), axis=1), axis=1)
ys = np.argmax(np.argmax(prediction.reshape(X.shape[0], h, w), axis=2), axis=1)

# TODO: fix color
plot_mps(fig, ax, xs, ys, data[:,0], np.random.rand(X.shape[0]))
plt.show()

#Simple SOFM for UK
plt.style.use('ggplot')

# Load data
data = load_uk_data().to_numpy()
X = data[:,1:]
print(X)

inp = X.shape[1]   # No of features (bill count)
h = 30            # Grid height
w = 30            # Grid width
rad = 3            # Neighbour radius
ep = 100           # No of epochs

# Create SOFM
sofmnet = algorithms.SOFM(
    n_inputs=inp,
    step=0.5,
    show_epoch=20,
    shuffle_data=True,
    verbose=True,
    learning_radius=rad,
    features_grid=(h,w),
)

sofmnet.train(X, epochs=ep)

#Visualizing Output
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(*sofmnet.weight, label='SOFM Weights')
ax.scatter3D(*X.T, label='Input');

ax.set_xlabel('vote_0')
ax.set_ylabel('vote_1')
ax.set_zlabel('vote_2')
ax.legend()

plt.show()