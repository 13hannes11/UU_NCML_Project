#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from neupy import algorithms
from itertools import product
import matplotlib.pyplot as plt

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
    plot_mps(data[:,0], xs, ys, data[:,1])
    plt.show()

    # plotting parties
    plot_parties(xs, ys, data[:,1])
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


def plot_parties(xs, ys, party_affiliation):
    cmap = plt.cm.RdYlGn
    # converting parties to numeric format 
    party_index_mapping, party_ids = np.unique(party_affiliation, return_inverse=True)

    # calculate average position of party
    party_count = np.zeros(party_index_mapping.shape[0])
    party_xs = np.zeros(party_index_mapping.shape[0])
    party_ys = np.zeros(party_index_mapping.shape[0])
    for x, y, party_id in zip(xs, ys, party_ids):
        party_xs[party_id] += x
        party_ys[party_id] += y
        party_count[party_id] += 1
    party_xs /= party_count
    party_ys /= party_count
    
    plt.figure()
    party_colors=np.array(range(len(party_index_mapping)))
    plt.scatter(party_xs, party_ys, c=party_colors, cmap=cmap)
    # plotting labels
    offset = 0.01
    for x,y, party in zip(party_xs, party_ys, party_index_mapping):
        plt.text(x + offset, y + offset, party)
