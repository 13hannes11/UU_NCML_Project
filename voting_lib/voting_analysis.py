#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from neupy import algorithms
from itertools import product
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn

def train_model(X, grid_h, grid_w, radius, step, ep):

    inp = X.shape[1]   # No of features (bills)

    # Create SOFM
    sofmnet = algorithms.SOFM(
        n_inputs=inp,
        step=0.5,
        show_epoch=100,
        shuffle_data=True,
        verbose=True,
        learning_radius=radius,
        features_grid=(grid_h, grid_w)
    )

    sofmnet.train(X, epochs=ep)
    return sofmnet

def predict(model, data, grid_h, grid_w):

    X = data[:,2:]
    
    # predicting mp positions
    prediction = model.predict(X)
    print(f'prediction: {prediction}')

    # converting to x and y coordinates
    ys, xs = np.unravel_index(np.argmax(prediction, axis=1), (grid_h, grid_w))

    # plotting mps
    party_affiliation = data[:,1]
    plot_mps(data[:,0], xs, ys, party_affiliation)
    plt.show()

    # calculating party positions based on mps

    party_pos = calc_party_pos(np.column_stack((xs, ys)), party_affiliation)

    # plotting parties
    plot_parties(party_pos)
    plt.show()

    # plotting party distances in outputspace
    part_distance_out = calc_party_distances(party_pos) 
    plot_party_distances(part_distance_out)
    plt.show()
    
    # Heatmap of weights
    plt.figure()
    weight = model.weight.reshape((model.n_inputs, grid_h, grid_w))
    heatmap = compute_heatmap(weight, grid_h, grid_w)
    plt.imshow(heatmap, cmap='Greys_r', interpolation='nearest')
    plt.axis('off')
    plt.colorbar()
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

def plot_hoverscatter(x, y, labels, colors, cmap = plt.cm.RdYlGn):
    fig,ax = plt.subplots()
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

def plot_mps(names, xs, ys, party_affiliation):
    # converting parties to numeric format 
    party_index_mapping, party_ids = np.unique(party_affiliation, return_inverse=True)

    # add random offset to show points that are in the same location
    ys_disp = ys + np.random.rand(ys.shape[0])
    xs_disp = xs + np.random.rand(xs.shape[0])
    parties = party_index_mapping[party_ids]
    plot_hoverscatter(xs_disp, ys_disp, names + " (" + parties + ")", party_ids)


def calc_party_pos(members_of_parliament, party_affiliation):
    party_index_mapping, party_ids = np.unique(party_affiliation, return_inverse=True)

    party_pos = np.zeros((party_index_mapping.shape[0], members_of_parliament.shape[1]))
    party_count = np.zeros((party_index_mapping.shape[0], members_of_parliament.shape[1]))
    party_pos

    for i, mp in enumerate(members_of_parliament):
        party_index = party_ids[i]
        party_pos[party_index] += mp
        party_count[party_index] += 1

    party_pos /= party_count

    return pd.DataFrame(data=party_pos, index=party_index_mapping)
def plot_parties(parties):
    cmap = plt.cm.RdYlGn
    
    party_index_mapping = parties.index
    
    plt.figure()
    party_colors=np.array(range(len(party_index_mapping)))
    plt.scatter(parties[0].to_numpy() , parties[1].to_numpy(), c=party_colors, cmap=cmap)

    # plotting labels
    offset = 0.01
    for x,y, party in zip(parties[0], parties[1], party_index_mapping):
        plt.text(x + offset, y + offset, party)
def calc_party_distances(parties):
    distances = np.zeros((parties.shape[0], parties.shape[0]))
    for i, (_, left_party) in enumerate(parties.iterrows()):
        for j, (_, top_party) in enumerate(parties.iterrows()):
            distances[i,j] = np.linalg.norm(left_party.to_numpy() - top_party.to_numpy())

    party_index_mapping = parties.index
    return pd.DataFrame(data=distances, index=party_index_mapping, columns=party_index_mapping)

def plot_party_distances(distances):
    fig = plt.figure()
    ax = plt.gca()
    ax.tick_params(axis="x", bottom=False, top=True, labelbottom=False, labeltop=True)
    sn.heatmap(distances, cmap='Oranges', annot=True)
