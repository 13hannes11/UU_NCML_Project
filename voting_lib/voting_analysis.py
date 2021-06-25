#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from neupy import algorithms
from itertools import product
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import re
from matplotlib.colors import ListedColormap

def train_model(X, grid_h, grid_w, radius, step, ep):

    inp = X.shape[1]   # No of features (bills)

    # Create SOFM
    sofmnet = algorithms.SOFM(
        n_inputs=inp,
        step=step,
        show_epoch=100,
        shuffle_data=True,
        verbose=True,
        learning_radius=radius,
        features_grid=(grid_h, grid_w)
    )

    sofmnet.train(X, epochs=ep)
    return sofmnet

def predict(model, data, grid_h, grid_w, party_colors, comparison_data=pd.DataFrame()):
    # default tight layout
    plt.rcParams["figure.autolayout"] = True

    X = data[:,2:]
    
    # predicting mp positions
    prediction = model.predict(X)
    print(f'prediction: {prediction}')

    # Plot hit map
    plot_hits(prediction, grid_w, grid_h)
    
    # converting to x and y coordinates
    ys, xs = np.unravel_index(np.argmax(prediction, axis=1), (grid_h, grid_w))

    # plotting mps
    party_affiliation = data[:,1]
    plot_mps(data[:,0], xs, ys, party_affiliation, party_colors, randomize_positions=True)
    plt.show()    

    # calculating party positions based on mps
    party_pos = calc_party_pos(np.column_stack((xs, ys)), party_affiliation)

    print(party_pos)

    # Plot node distnaces
    plt.figure()
    weight = model.weight.reshape((model.n_inputs, grid_h, grid_w))
    heatmap = compute_heatmap(weight, grid_h, grid_w)
    plt.imshow(heatmap, cmap ='Blues', interpolation='nearest',zorder=1, alpha=0.5)
    plt.axis('off')
    plt.colorbar()
    
    # plotting parties
    plot_parties(party_pos, party_colors, randomize_positions=False, new_plot=False)

    # plotting party distances in output space
    part_distance_out = calc_party_distances(party_pos) 
    plot_party_distances(part_distance_out)
    plt.show()

    if not comparison_data.empty:
       plot_parties(comparison_data, party_colors, randomize_positions=False, new_plot=True)
       plt.ylabel("libertarian - authoritarian")
       plt.xlabel("left < economic > right")
              
       comparison_data_dist = calc_party_distances(comparison_data)
       plot_party_distances(comparison_data_dist)

       err = remove_NaN_rows_columns(normalize_df(part_distance_out) - normalize_df(comparison_data_dist))
       err = err * err
       plot_party_distances(err)
       plt.title(f'MSE={np.nanmean(err.to_numpy()):.2f}')
       plt.show()

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

def plot_hoverscatter(x, y, categories, hover_labels, colors, cmap = plt.cm.RdYlGn):
    fig, ax = plt.subplots()
    ANNOTATION_DISTANCE = 5
    TRANSPARENCY = 0.8
    scatterplot = plt.scatter(x,y,c=colors, s=5, cmap=cmap)

    handles, labels = scatterplot.legend_elements(prop="colors", alpha=0.6)
    print(labels[0])
    cat = list(map(lambda l: categories[int(re.sub(r'([^\d]+)', "", l))], labels))
    
    legend = ax.legend(handles, cat, bbox_to_anchor=(1.3, 1), loc='upper left')

    annot = ax.annotate("", xy=(0,0), 
                        xytext=(ANNOTATION_DISTANCE, ANNOTATION_DISTANCE),
                        textcoords="offset points",
                        bbox=dict(boxstyle="Square"))
    annot.set_visible(False)

    def update_annot(ind):
        index = ind["ind"][0]
        pos = scatterplot.get_offsets()[index]
        annot.xy = pos
        text = f'{hover_labels[index]}'
        annot.set_text(text)
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

def plot_mps(names, xs, ys, party_affiliation, party_colors, randomize_positions=True):
    # converting parties to numeric format 
    party_index_mapping, party_ids = np.unique(party_affiliation, return_inverse=True)

    # add random offset to show points that are in the same location
    if randomize_positions:
        xs_disp = xs + np.random.rand(xs.shape[0]) - 0.5
        ys_disp = ys + np.random.rand(ys.shape[0]) - 0.5
    else:
        xs_disp = xs
        ys_disp = ys
    
    parties = party_index_mapping[party_ids]

    colormap = ListedColormap(list(map(lambda x: party_colors[x], party_index_mapping)))
    plot_hoverscatter(xs_disp, ys_disp, party_index_mapping, names + " (" + parties + ")", party_ids, cmap=colormap)

def calc_party_pos(members_of_parliament, party_affiliation):
    party_index_mapping, party_ids = np.unique(party_affiliation, return_inverse=True)

    party_pos = np.zeros((party_index_mapping.shape[0], members_of_parliament.shape[1]))
    party_count = np.zeros((party_index_mapping.shape[0], members_of_parliament.shape[1]))

    for i, mp in enumerate(members_of_parliament):
        party_index = party_ids[i]
        party_pos[party_index] = party_pos[party_index] + mp
        party_count[party_index] += 1

    party_pos /= party_count

    return pd.DataFrame(data=party_pos, index=party_index_mapping)

def plot_parties(parties, party_colors, randomize_positions=False, new_plot=True):
    
    party_index_mapping = parties.index

    colors = list(map(lambda x: party_colors[x], party_index_mapping))

    
    if new_plot:
        plt.figure()
    
    if randomize_positions:
        xs_disp = parties[0].to_numpy() + np.random.rand(parties.shape[0]) - 0.5
        ys_disp = parties[0].to_numpy() + np.random.rand(parties.shape[0]) - 0.5
    else:
        xs_disp = parties[0].to_numpy() 
        ys_disp = parties[1].to_numpy()
    
    for i, party in enumerate(party_index_mapping):
        print("Party ", party, " x = ", xs_disp[i], "y = ", ys_disp[i])
        plt.scatter(xs_disp[i], ys_disp[i], label=party, zorder=2, c=colors[i], edgecolors='None')
        
    plt.legend(title='Parties', bbox_to_anchor=(1.3, 1), loc='upper left')    
    
def calc_party_distances(parties):
    distances = np.zeros((parties.shape[0], parties.shape[0]))
    for i, (_, left_party) in enumerate(parties.iterrows()):
        for j, (_, top_party) in enumerate(parties.iterrows()):
            distances[i,j] = np.linalg.norm(left_party.to_numpy() - top_party.to_numpy())

    party_index_mapping = parties.index
    return pd.DataFrame(data=distances, index=party_index_mapping, columns=party_index_mapping)

def plot_party_distances(distances):
    plt.figure()
    ax = plt.gca()
    ax.tick_params(axis="x", bottom=False, top=True, labelbottom=False, labeltop=True)
    sn.heatmap(distances, cmap='Oranges', annot=True)

def plot_hits(prediction, grid_w, grid_h):    
    hits = (prediction.sum(axis=0)).reshape(grid_w, grid_h)
    plt.figure()
    sn.heatmap(hits, annot=True, xticklabels=False, yticklabels=False, cbar=False)
    
def normalize_df(dataframe):
    df = dataframe.copy(deep=True)
    df = df - np.min(df.to_numpy())
    df = df / np.max(df.to_numpy()) 
    return df

def remove_NaN_rows_columns(dataframe):
    df = dataframe.copy(deep=True)
    df = df.dropna(axis=0, how='all', thresh=None, subset=None, inplace=False)
    df = df.dropna(axis=1, how='all', thresh=None, subset=None, inplace=False)
    return df
