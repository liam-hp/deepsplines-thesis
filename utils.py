from tabulate import tabulate
from datetime import datetime, timedelta
import numpy as np
import math, json
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
from itertools import accumulate
import linspline
from models import LinearReLU, LinearBSpline
from IPython.display import clear_output
import pandas as pd
import seaborn as sns
import utils, importlib
from models import LinearReLU, LinearBSpline
import linspline
from sklearn.datasets import fetch_california_housing
import time, torch
from deepspeed.profiling.flops_profiler import get_model_profile
from torch.utils.data import TensorDataset, DataLoader

# from deepspeed.profiling.flops_profiler import get_model_profile #! i think this slows down all runs
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import sys, os
sys.path.insert(0, os.path.abspath('../DeepSplines'))
import deepsplines

def time_str_to_timedelta(time_str):
    return datetime.strptime(time_str, "%H:%M:%S.%f") - datetime(1900, 1, 1)

# Convert timedelta back to time string
def timedelta_to_time_str(td):
    total_seconds = int(td.total_seconds())
    ms = td.microseconds // 10000
    return f"{total_seconds // 3600}:{(total_seconds % 3600) // 60}:{total_seconds % 60}.{ms:02d}"

def display_vals_table(data):

  best_loss_vals = [row[1] for row in data]
  final_loss_vals = [row[2] for row in data]
  time_deltas = [time_str_to_timedelta(row[3]) for row in data]
  fwd_latency_vals = [row[4] for row in data]

  average_best_loss = round(np.mean(best_loss_vals), 4)
  average_final_loss = round(np.mean(final_loss_vals), 4)
  average_time_delta = sum(time_deltas, timedelta()) / len(time_deltas)
  average_time = timedelta_to_time_str(average_time_delta)
  average_fwd_latency = round(np.mean(fwd_latency_vals), 4)

  data.append(["-", "-", "-", "-", "-", "-"])
  data.append(["Avg", average_best_loss, average_final_loss, average_time, average_fwd_latency, "n/a"])

  headers = ['Run #', 'Best Loss', 'Final Loss', 'Training Time', 'Fwd Latency', 'Outlier?']
  print(tabulate(data, headers=headers, tablefmt="pipe"))

  print("\nBest loss STD: " + str(np.std(final_loss_vals)))
  print("Final loss STD: " + str(np.std(best_loss_vals)) + "\n")

def display_vals_plot(data, timing=[], ax=None):

  with_times = len(timing) > 0
  colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan', 'magenta']

  for i in range(len(data)):
    if(not with_times):
      ax.plot(data[i], label=f'Validation Loss {i}', color=colors[i % len(colors)], alpha=.1, lw=1)
    else:
      ax.plot(timing[i], data[i], label=f'Validation Loss {i}', color=colors[i % len(colors)], alpha=.1, lw=1)

  average_losses = np.mean(np.array(data), axis=0)
  average_times = np.mean(np.array(timing), axis=0)

  if(with_times):
    ax.plot(average_times, average_losses, label='Average Validation Loss', color='black', lw=1)
    
    for j in list(np.linspace(0, len(average_losses) - 1, 10, dtype=int)) + [len(average_losses)-1]:
      ax.text(
          average_times[j],
          min(average_losses[j] + 0.1, 3),
          f'{average_losses[j]:.2f}',
          ha='center',
          va='bottom',
          fontsize=8
      )
      ax.plot([average_times[j], average_times[j]], [average_losses[j] - 0.1, average_losses[j] + 0.1],  # Short vertical segment
              color='black', linestyle='--', linewidth=1)
  
  else:
    ax.plot(average_losses, label='Average Validation Loss', color='black', lw=1)

    for j in list(np.linspace(0, len(average_losses) - 1, 10, dtype=int)) + [len(average_losses)-1]:
      ax.text(
        j,
        min(average_losses[j] + 0.1, 3),
        f'{average_losses[j]:.2f}',
        ha='center',
        va='bottom',
        fontsize=8
      )
      ax.plot([j, j], [average_losses[j] - 0.1, average_losses[j] + 0.1],  # Short vertical segment
          color='black', linestyle='--', linewidth=1)
  
  ax.grid(axis='y')

def plot_single(title, run_vals, run_times):
    fig, (ax1, ax2) = plt.subplots(2)
    # ax1.set_title("Val Loss over ")
    # ax2.set_title("Title for Ax2")
    ax1.set_xlabel("Epochs", labelpad=20)  # Adjust labelpad for more space
    ax1.set_ylabel("Val Loss", labelpad=20)

    ax2.set_xlabel("Time (s)", labelpad=20)
    ax2.set_ylabel("Val Loss", labelpad=20)

    ax1.set_ylim(0, 3)
    ax2.set_ylim(0, 3)

    plt.subplots_adjust(hspace=.7)

    display_vals_plot(run_vals, [], ax1)
    display_vals_plot(run_vals, run_times, ax2)

    fig.suptitle(title)
    fig.show()

def plot_single2(model_path, x="time"):
    with open(f'saved/{model_path[0]}.json', 'r') as file:
        model_data = json.load(file)

    values = np.array(model_data[model_path[1]])
    run_times = np.array(model_data['times'])
    cumul_times = list(accumulate(run_times))

    fig, (ax1, ax2) = plt.subplots(2)

    ax1.set_xlabel("Epochs", labelpad=20)  # Adjust labelpad for more space
    ax1.set_ylabel("Val Loss", labelpad=20)

    ax2.set_xlabel("Time (s)", labelpad=20)
    ax2.set_ylabel("Val Loss", labelpad=20)

    ax1.set_ylim(0, 3)
    ax2.set_ylim(0, 3)

    plt.subplots_adjust(hspace=.7)

    display_vals_plot(values, [], ax1)
    display_vals_plot(values, cumul_times, ax2)

    fig.show()

def generate_colors(n, opacity=1):
      cmap = plt.cm.rainbow
      colors = cmap(np.linspace(0, 1, n))
      colors[:, 3] = opacity
      return colors

def plot_bspline(locs, coeffs, degree=1, scale_by_coeff=False, hide_bases=False):
    
    def relu(x):
      return np.maximum(0, x)

    colors = generate_colors(len(coeffs))

    # is this right??
    knots = np.concatenate((
        [locs[0]] * (degree + 1),  # Repeated knots at the start
        locs[1:-1],               # Interior knot
        [locs[-1]] * (degree + 1)  # Repeated knots at the end
    ))

    bspline = BSpline(knots, coeffs, degree)
    
    if(hide_bases):
        fig, ax1 = plt.subplots(figsize=(8, 6))
    else:    
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

    x = np.linspace(min(locs), max(locs), 100)
    y = relu(x)
    ax1.plot(x, y, 'grey', label='ReLU')

    ax1.plot(x, bspline(x), 'black', lw=2, alpha=0.7, label='BSpline')
    for i, (x, y) in enumerate(zip(locs, coeffs)):
        ax1.scatter(x, y, color=colors[i], s=50, zorder=5, label='Control Points')
        ax1.text(x, y, f'$P_{{{i + 1}}}$', color=colors[i], ha='right', va='bottom')
    ax1.grid(True)

    ax1.set_title('B-Spline and Control Points')
    ax1.legend()

    if(not hide_bases):
        n_basis = len(knots) - (degree + 1)
        x = np.linspace(knots[0], knots[-1], 1000)

        for i in range(n_basis):
            # Coefficients for the i-th basis function: all zeros except for a 1 at position i
            coeffs_basis = np.zeros(n_basis)
            coeffs_basis[i] = 1
            
            # Create the i-th basis function
            spline = BSpline(knots, coeffs_basis, degree)
            
            # Evaluate the basis function
            y = spline(x)
            
            # Plot the basis function
            if(scale_by_coeff):
                ax2.plot(x, coeffs[i]*y, label=f'$B_{i}(x)$', alpha=0.75, color=colors[i])
                ax2.set_title('B-Spline Basis Functions (Scaled by Coeffs)')
            else: 
                ax2.plot(x, y, label=f'$B_{i}(x)$', alpha=0.75, color=colors[i])
                ax2.set_title('B-Spline Basis Functions')
        
        ax2.grid(True)

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline

def plot_bspline_sb_side_by_side(locs, coeffs, degree=1, scale_by_coeff=False, hide_bases=False):
    """
    Plot a B-spline using Seaborn for styling and color palette, with an option to show
    the spline on one subplot and the basis functions on another subplot, side by side.

    Parameters
    ----------
    locs : array-like
        The x-locations of the control points (and boundary knots).
    coeffs : array-like
        The coefficients (or y-values) corresponding to `locs`.
    degree : int, optional
        The degree of the B-spline, by default 1.
    scale_by_coeff : bool, optional
        If True, each basis function is scaled by the coefficient value. Default is False.
    hide_bases : bool, optional
        If True, only show the main B-spline and control points (no basis functions).
        Default is False.
    """
    # Use a Seaborn theme and automatically chosen color palette
    sns.set_theme(style="whitegrid")

    # Simple ReLU function for reference
    def relu(x):
        return np.maximum(0, x)
    
    # Generate colors from Seaborn palette
    colors = sns.color_palette("husl", n_colors=len(coeffs))

    # Construct the knots
    knots = np.concatenate((
        [locs[0]] * (degree + 1),  # repeated at start
        locs[1:-1],               # interior
        [locs[-1]] * (degree + 1) # repeated at end
    ))
    
    # Create the BSpline object
    bspline = BSpline(knots, coeffs, degree)
    
    # Set up the figure
    if hide_bases:
        # Only one subplot if we're hiding the basis functions
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
        ax2 = None
    else:
        # Two subplots, side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot ReLU as a reference on ax1
    x_vals = np.linspace(min(locs), max(locs), 100)
    y_relu = relu(x_vals)
    ax1.plot(x_vals, y_relu, color='grey', label='ReLU')
    
    # Plot the main B-spline
    ax1.plot(x_vals, bspline(x_vals), 'black', lw=2, alpha=0.7, label='BSpline')
    
    # Plot control points
    for i, (x_pt, y_pt) in enumerate(zip(locs, coeffs)):
        ax1.scatter(x_pt, y_pt, color=colors[i], s=50, zorder=5)
        ax1.text(x_pt, y_pt, f'$P_{{{i+1}}}$', color=colors[i],
                 ha='right', va='bottom', fontsize=9)
    
    ax1.set_title('B-Spline with Control Points')
    ax1.legend()
    
    # If bases are not hidden, plot them on ax2
    if not hide_bases:
        # Number of basis functions
        n_basis = len(knots) - (degree + 1)
        
        # Fine grid for evaluating the basis functions
        x_fine = np.linspace(knots[0], knots[-1], 1000)
        
        for i in range(n_basis):
            # Build the i-th basis function (all zeros except 1 at index i)
            coeffs_basis = np.zeros(n_basis)
            coeffs_basis[i] = 1
            spline_i = BSpline(knots, coeffs_basis, degree)
            
            y_basis = spline_i(x_fine)
            
            # Either scale the basis function by its coefficient or not
            if scale_by_coeff:
                ax2.plot(x_fine, coeffs[i] * y_basis, 
                         label=f'$B_{i}(x)$ (scaled)', alpha=0.75, color=colors[i])
                ax2.set_title('B-Spline Basis Functions (Scaled by Coefficients)')
            else:
                ax2.plot(x_fine, y_basis,
                         label=f'$B_{i}(x)$', alpha=0.75, color=colors[i])
                ax2.set_title('B-Spline Basis Functions')
        
        ax2.legend()
        ax2.grid(True)
    
    plt.tight_layout()
    plt.show()




def plot_linspline(locs, coeffs, degree=1, scale_by_coeff=True):
    
    def relu(x):
      return np.maximum(0, x)

    colors = generate_colors(len(coeffs))

    # is this right??
    knots = np.concatenate((
        [locs[0]] * (degree + 1),  # Repeated knots at the start
        locs[1:-1],               # Interior knots
        [locs[-1]] * (degree + 1)  # Repeated knots at the end
    ))

    bspline = BSpline(knots, coeffs, degree)
    
    fig, ax1 = plt.subplots(2, 1, figsize=(8, 10))

    x = np.linspace(min(locs), max(locs), 100)
    y = relu(x)
    ax1.plot(x, y, 'grey', label='ReLU')
    ax1.legend()

    ax1.plot(x, bspline(x), 'black', lw=2, alpha=0.7, label='BSpline')
    for i, (x, y) in enumerate(zip(locs, coeffs)):
        ax1.scatter(x, y, color=colors[i], s=50, zorder=5, label='Control Points')
        ax1.text(x, y, f'$P_{{{i + 1}}}$', color=colors[i], ha='right', va='bottom')
    ax1.grid(True)
    ax1.set_title('B-Spline and Control Points')

def plot_layer(activation_layer, num_activ_per_plot):
    activ_name = activation_layer['name']
    sparsity_mask = activation_layer['sparsity_mask'].numpy()
    num_units, size = activation_layer['locations'].size()

    # Assumes that all activations have the same range/#coefficients
    locations = activation_layer['locations'][0].numpy()
    coefficients = activation_layer['coefficients'].numpy()

    div = coefficients.shape[0] * 1. / num_activ_per_plot
    # number of plots for this activation layer
    total = int(np.ceil(div))
    # number of plots with num_activ_per_plot activations
    quotient = int(np.floor(div))
    # number of activations in the last plot
    remainder = coefficients.shape[0] - quotient * num_activ_per_plot

    for j in range(total):
        # plot half dashed and half full
        plt.figure()
        ax = plt.gca()
        ax.grid()

        # start/end indexes of activations to plot in this layer
        start_k = j * num_activ_per_plot
        end_k = start_k + num_activ_per_plot
        if remainder != 0 and j >= total - 1:
            end_k = start_k + remainder

        for k in range(start_k, end_k):
            ax.plot(locations, coefficients[k, :], linewidth=1.0)

            # if args.plot_sparsity:
            #     ls = matplotlib.rcParams['lines.markersize']
            #     non_sparse_relu_slopes = (sparsity_mask[k, :])
            #     # relu slopes locations range from the second (idx=1) to
            #     # second to last (idx=-1) B-spline coefficients
            #     ax.scatter(locations[1:-1][non_sparse_relu_slopes],
            #                 coefficients[k, 1:-1][non_sparse_relu_slopes],
            #                 s=2 * (ls**2))

            #     sparse_relu_slopes = (sparsity_mask[k, :] is False)
            #     ax.scatter(locations[1:-1][sparse_relu_slopes],
            #                 coefficients[k, 1:-1][sparse_relu_slopes],
            #                 s=2 * (ls**2))

        x_range = ax.get_xlim()
        assert x_range[0] < 0 and x_range[1] > 0, f'x_range: {x_range}.'
        y_tmp = ax.get_ylim()
        assert y_tmp[0] < y_tmp[1], f'y_tmp: {y_tmp}.'

        y_range = x_range  # square axes by default
        if y_tmp[0] < x_range[0]:
            y_range[0] = y_tmp[0]
        if y_tmp[1] > x_range[1]:
            y_range[1] = y_tmp[1]

        ax.set_ylim([*y_range])
        ax.set_xlabel(r"$x$", fontsize=20)
        ax.set_ylabel(r"$\sigma(x)$", fontsize=20)

        title = activ_name + f'_neurons_{start_k}_{end_k}'
        ax.set_title(title, fontsize=20)

        # plt.subplots_adjust(hspace = 0.4)
        plt.show()

def calc_bspline_flops(model):
  
  '''
    FLOPS_per_spline = d_in x d_out x [9 x K x (G + 1.5 x K) + 2 × G − 2.5 × K + 3]

    We use K to denote the order of the spline, which corresponds to the parameter k of the official nn.Module
        KANLayer. It is the order of the Polynomial basis in the spline function. We use G to denote
        the number of spline intervals, which corresponds the num parameter of the official nn.Module
        KANLayer. It is the number of intervals of the B-spline, before padding. It equals the number of
        control points - 1, before padding. After padding, there should be (K +G) functioning control points. - Fairer
  '''

  
  d_in = 1 # MLP sums before activation
  d_out = 1 # out is also just one result that is sent to all nodes in the next layer

  # degree
  K = 1 

  fwd_spline_flops = 0
  for layer in model.get_deepspline_activations(): # for each layer
      num_activations = len(layer['locations']) # the number of bsplines on that layer
      G = len(layer['locations'][0])-1 # number of control points per activation
      # all activations are identical, so we can just multiply through
      fwd_spline_flops += (d_in * d_out * (9*K * (G + 1.5 * K) + 2 * G - 2.5 * K + 3)) * num_activations

  return fwd_spline_flops

def calc_lspline_flops(model):
    # FLOPS estimation based on implementation
    flops = 0
    for layer in model.get_layers():
        if(isinstance(layer, linspline.LinearSplineLayer)):
            flops += layer.get_flops()
    return flops

def point_dist(p1, p2):
    x1, y1 = p1
    x2, y2 = p2

    dist = math.sqrt(math.pow((x1-x2),2) + math.pow((y1*1000-y2*1000),2)) # scale y to account for diff btw x and y axes
    return dist

def plot_multiple(data, model_selector_pairs, x="time"):
    fig, ax = plt.subplots()

    colors = generate_colors(len(model_selector_pairs))

    label_space = []

    for i, (model,selector) in enumerate(model_selector_pairs):
        run_vals = data[model][selector]['vals']
        run_times = data[model][selector]['times']

        average_losses = np.mean(np.array(run_vals), axis=0)
        average_times = np.mean(np.array(run_times), axis=0)

        if(x=="time"):
            ax.plot(average_times, average_losses, label=f'{model}: {selector}', color=colors[i], lw=1)
        else:
            ax.plot(average_losses, label=f'{model}: {selector}', color=colors[i], lw=1)
        
        labelx = average_times[-1], # x position
        labely = average_losses[-1], # y position

        labelx = labelx[0]
        labely = labely[0] + .01

        for l in label_space:
            while(point_dist((l), (labelx, labely)) < 30):
                labely += .005
        
        label_space.append((labelx, labely))

        ax.text(
            labelx if x=="time" else len(average_losses)-1,
            min(1, labely),
            f'{int(average_times[-1])}s:{average_losses[-1]:.03f}' if x=="time" else f'{average_losses[-1]:.03f}' , # dislay text
            color="black", #colors[i], 
            fontsize=8, 
            ha='center', 
            va='bottom'
        )

    
    ax.set_ylabel("Val Loss", labelpad=10)
    ax.set_xlabel("Time (s)" if x=="time" else "Epochs", labelpad=10)
    ax.set_ylim(.3, 1)
    ax.legend()

def plot_multiple2(model_paths, x="time", xlim=None, ylim=None, xmin=None, ymin=None, vbars=[], hide_legend=False, ):
    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(10, 5),
        gridspec_kw={'width_ratios': [3, 1]}
    )

    colors = generate_colors(len(model_paths))
    table_data = []

    for i, m in enumerate(model_paths):
        with open(f'saved/{m[0]}/{m[1]}.json', 'r') as file:
            model_data = json.load(file)

        values = np.array(model_data[m[2]])
        
        if(m[2][:2] == "cr"):
            run_times = np.array(model_data['cr_times'])
        else:
            run_times = np.array(model_data['times']) # this is timing per epoch per run [r1, r2, r3, ..] w/ r1=[e1, e2, e3, ...]
        
        plot_times = np.mean(run_times, axis=0) # average over runs: [e1, e2, e3, ...]

        cumul_times = list(accumulate(plot_times))# cumulative sum at each point
        # print(plot_times)
        # print(cumul_times)
        
        if m[3] == "avg":
            plot_losses = np.mean(values, axis=0)
        elif m[3] == "median":
            plot_losses = np.median(values, axis=0)
        else:  # best
            plot_losses = np.min(values, axis=0)


        if x == "time":
            ax1.plot(
                cumul_times, plot_losses,
                label=f'{m[1]},{m[2]},{m[3]}',
                color=colors[i], lw=1, alpha=.8
            )
        else:
            ax1.plot(
                plot_losses,
                label=f'{m[1]},{m[2]},{m[3]}',
                color=colors[i], lw=1, alpha=.8
            )

        # Append only the final loss to the table
        table_data.append([f"{plot_losses[-1]:.03f}"])
        
    ax1.set_ylabel("Val Loss", labelpad=10)
    ax1.set_xlabel("Time (s)" if x == "time" else "Epochs", labelpad=10)
    
    if ylim is not None:
        ax1.set_ylim(0 if ymin==None else ymin, ylim)
    if xlim is not None:
        ax1.set_xlim(0 if xmin==None else xmin, xlim)

    for (loc, col) in vbars:
        ax1.axvline(x=loc, color=col, linestyle='--')
    


    ax2.axis('off')
    tbl = ax2.table(
        cellText=table_data,
        colLabels=["Final Loss"],
        loc='center'
    )
    tbl.scale(1.5, 1.5)

    if(not hide_legend):
        ax1.legend()

    for i, row in enumerate(table_data):
        tbl[(i+1, 0)]._text.set_color(colors[i])

    for (r, c), cell in tbl.get_celld().items():
        cell.set_text_props(ha='center', va='center')

def plot_multiple_3(model_paths, model_names=None, title=None, x="time", y_ax="Val Loss", xlim=None, ylim=None, xmin=None, ymin=None, vbars=[], hide_legend=False):
    
    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(10, 5),
        gridspec_kw={'width_ratios': [3, 1]}
    )
    
    if(title):
        fig.suptitle(title)

    colors = generate_colors(len(model_paths))
    table_data = []

    for i, m in enumerate(model_paths):
        with open(f'saved/{m[0]}/{m[1]}.json', 'r') as file:
            model_data = json.load(file)

        values = np.array(model_data[m[2]])
        
        if(m[2][:2] == "cr"):
            run_times = np.array(model_data['cr_times'])
        else:
            run_times = np.array(model_data['times']) # this is timing per epoch per run [r1, r2, r3, ..] w/ r1=[e1, e2, e3, ...]
        
        plot_times = np.mean(run_times, axis=0) # average over runs: [e1, e2, e3, ...]

        cumul_times = list(accumulate(plot_times))# cumulative sum at each point
        # print(plot_times)
        # print(cumul_times)
        
        if m[3] == "avg":
            plot_losses = np.mean(values, axis=0)
        elif m[3] == "median":
            plot_losses = np.median(values, axis=0)
        else:  # best
            plot_losses = np.min(values, axis=0)


        if(model_names == None):
            model_name = f'{m[1]},{m[2]},{m[3]}'
        else:
            model_name = model_names[i]
        
        if x == "time":
            ax1.plot(
                cumul_times, plot_losses,
                label=model_name,
                color=colors[i], lw=1, alpha=.8
            )
        else:
            ax1.plot(
                plot_losses,
                label=model_name,
                color=colors[i], lw=1, alpha=.8
            )

        # Append only the final loss to the table
        table_data.append([model_name, f"{plot_losses[-1]:.03f}"])
        
    ax1.set_ylabel(y_ax, labelpad=10)
    ax1.set_xlabel("Time (s)" if x == "time" else "Epochs", labelpad=10)
    
    if ylim is not None:
        ax1.set_ylim(0 if ymin==None else ymin, ylim)
    if xlim is not None:
        ax1.set_xlim(0 if xmin==None else xmin, xlim)

    for (loc, col) in vbars:
        ax1.axvline(x=loc, color=col, linestyle='--')
    
    if(not hide_legend):
        ax1.legend()

    ax2.axis('off')
    tbl = ax2.table(
        cellText=table_data,
        colLabels=["Model", "Final Loss"],
        loc='center'
    )

    tbl.scale(1.5, 1.5)

    # table styling
    for i, row in enumerate(table_data):
        tbl[(i+1, 0)]._text.set_color(colors[i])
        tbl[(i+1, 1)]._text.set_color(colors[i])
    for (r, c), cell in tbl.get_celld().items():
        cell.set_text_props(ha='center', va='center')

def plot_multiple_4(model_paths, aes=None, title=None, x="time", y_ax="Val Loss", xlim=None, ylim=None, xmin=None, ymin=None, vbars=[], hide_legend=False):
    
    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(10, 5),
        gridspec_kw={'width_ratios': [3, 1]}
    )
    
    if(title):
        fig.suptitle(title)

    table_data = []

    for i, m in enumerate(model_paths):
        with open(f'saved/{m[0]}/{m[1]}.json', 'r') as file:
            model_data = json.load(file)

        values = np.array(model_data[m[2]])
        
        if(m[2][:2] == "cr"):
            run_times = np.array(model_data['cr_times'])
        else:
            run_times = np.array(model_data['times']) # this is timing per epoch per run [r1, r2, r3, ..] w/ r1=[e1, e2, e3, ...]
        
        plot_times = np.mean(run_times, axis=0) # average over runs: [e1, e2, e3, ...]

        cumul_times = list(accumulate(plot_times))# cumulative sum at each point
        # print(plot_times)
        # print(cumul_times)
        
        if m[3] == "avg":
            plot_losses = np.mean(values, axis=0)
        elif m[3] == "median":
            plot_losses = np.median(values, axis=0)
        else:  # best
            plot_losses = np.min(values, axis=0)


        line_style="-"
        if(aes == None):
            model_name = f'{m[1]},{m[2]},{m[3]}'
        else:
            model_name = aes[i][0]
            line_style=aes[i][1]
            color=aes[i][2]
        
        if x == "time":
            ax1.plot(
                cumul_times, plot_losses,
                label=model_name,
                color=color, 
                linestyle=line_style, 
                lw=1, alpha=.8
            )
        else:
            ax1.plot(
                plot_losses,
                label=model_name,
                color=color,
                linestyle=line_style,
                lw=1, alpha=.8
            )

        # Append only the final loss to the table
        table_data.append([model_name, f"{plot_losses[-1]:.03f}"])
        
    ax1.set_ylabel(y_ax, labelpad=10)
    ax1.set_xlabel("Time (s)" if x == "time" else "Epochs", labelpad=10)
    
    if ylim is not None:
        ax1.set_ylim(0 if ymin==None else ymin, ylim)
    if xlim is not None:
        ax1.set_xlim(0 if xmin==None else xmin, xlim)

    for (loc, col) in vbars:
        ax1.axvline(x=loc, color=col, linestyle='--')
    
    if(not hide_legend):
        ax1.legend()

    ax2.axis('off')
    tbl = ax2.table(
        cellText=table_data,
        colLabels=["Model", "Final Loss"],
        loc='center'
    )

    tbl.scale(1.5, 1.5)

    # table styling
    for i, row in enumerate(table_data):
        tbl[(i+1, 0)]._text.set_color(aes[i][2])
        tbl[(i+1, 1)]._text.set_color(aes[i][2])
    for (r, c), cell in tbl.get_celld().items():
        cell.set_text_props(ha='center', va='center')

def plot_seaborn(
        model_paths, aes=None, 
        palette="bright", title=None, x="time", y_ax="MSE Loss", xlim=None, ylim=None, xmin=None, ymin=None, 
        aes1="Arch", aes2="Layers",
        vbars=[], hide_legend=False, cutoff=None):
    
    sns.set_theme(
        palette=palette,
        style="whitegrid",
    )

    # Data collection for plotting
    plot_data = []

    for i, m in enumerate(model_paths):
        with open(f'saved/{m[0]}/{m[1]}.json', 'r') as file:
            model_data = json.load(file)

        values = np.array(model_data[m[2]])
        
        if m[2][:2] == "cr":
            run_times = np.array(model_data['cr_times'])
        else:
            run_times = np.array(model_data['times'])

        plot_times = np.mean(run_times, axis=0)
        cumul_times = list(accumulate(plot_times))

        if m[3] == "avg":
            plot_losses = np.mean(values, axis=0)
        elif m[3] == "med":
            plot_losses = np.median(values, axis=0)
        elif m[3] == "best":
            plot_losses = np.min(values, axis=0)

        if(cutoff != None):
            plot_losses = plot_losses[:cutoff]
            cumul_times = cumul_times[:cutoff]

        print(f"Avg over last 5 epochs on {aes[i][0]},{aes[i][1]},{m[3]}: {np.mean(plot_losses[-5])} in {cumul_times[-1]}s")
        
        for t, l in zip((cumul_times if x == "time" else range(len(plot_losses))), plot_losses):
            plot_data.append({
                "Time" if x == "time" else "Epoch": t,
                y_ax: l,
                aes1: aes[i][0],
                aes2: aes[i][1],
            })

    # Convert to DataFrame
    df = pd.DataFrame(plot_data)

    # Plotting with Seaborn
    plt.figure(figsize=(10, 6))
    x_col = "Time" if x == "time" else "Epoch"

    ax = sns.lineplot(data=df, x=x_col, y=y_ax, hue=aes1, style=aes2, markers=False)

    if title:
        ax.set_title(title, fontsize=14, pad=15)
    
    ax.set_xlabel("Time (s)" if x == "time" else "Epochs", labelpad=10)
    ax.set_ylabel(y_ax, labelpad=10)

    if ylim is not None:
        ax.set_ylim(ymin if ymin is not None else 0, ylim)
    if xlim is not None:
        ax.set_xlim(xmin if xmin is not None else 0, xlim)

    # Add vertical bars
    for (loc, col) in vbars:
        plt.axvline(x=loc, color=col, linestyle="--")

    # Hide legend if needed
    ax.legend(loc="upper right")
    if hide_legend:
        ax.legend_.remove()

    # Display the plot
    plt.tight_layout()
    plt.show()

def run_layer(layer, input_dim, batch_size, num_inputs):
    #print(layer)
    start_time = time.perf_counter()
    inputs = [ torch.rand(batch_size, input_dim) * 3 - 1.5 for _ in range(num_inputs) ]
    # print("Example input: ", inputs[0])
    for i in range(num_inputs):
        _ = layer(inputs[i])
    return time.perf_counter() - start_time # time to process n inputs

def get_model_params(model, input_size=8, batch_size=10):
    base_flops, macs, params = get_model_profile(model=model, # model
            input_shape=(batch_size, input_size), # input shape to the model. If specified, the model takes a tensor with this shape as the only positional argument.
            args=None, # list of positional arguments to the model.
            kwargs=None, # dictionary of keyword arguments to the model.
            print_profile=False, #! prints the model graph with the measured profile attached to each module
            detailed=True, # print the detailed profile
            module_depth=-1, # depth into the nested modules, with -1 being the inner most modules
            top_modules=1, # the number of top modules to print aggregated profile
            warm_up=10, # the number of warm-ups before measuring the time of each module
            as_string=False, # print raw numbers (e.g. 1000) or as human-readable strings (e.g. 1k)
            output_file=None, # path to the output file. If None, the profiler prints to stdout.
            ignore_modules=None)

    return params

def graph_params_x_lat(archs):

    input_dim = 8
    batch_size = 10
    
    relu_results = []
    bspline_results = []
    lspline_results = []

    archs_pos = []

    for arch in archs:

        reluModel = LinearReLU(arch)
        # run_layer(reluModel.layers[1], input_dim, batch_size)
        relu_results.append((arch, run_layer(reluModel, input_dim, batch_size)))

        bsplineModel = LinearBSpline(arch, 3, 1, "relu")
        # spline_params = get_model_params(bsplineModel) #! not actually true... need to figure out why bspline has fewer params

        # archs_pos.append((arch, spline_params))

        # run_layer(bsplineModel.get_layers()[1], input_dim, batch_size)
        bspline_results.append((arch, run_layer(bsplineModel, input_dim, batch_size)))

        linModel = linspline.LSplineFromBSpline(bsplineModel.get_layers())
        # run_layer(linModel.get_layers()[1], input_dim, batch_size)
        lspline_results.append((arch, run_layer(linModel, input_dim, batch_size)))

    clear_output()

    # Assuming relu_results, bspline_results, and lspline_results contain (idx, result) tuples

    labels = [str(result[0]) for result in relu_results]
    relu_values = [result[1] for result in relu_results]

    bspline_values = [result[1] for result in bspline_results]

    lspline_values = [result[1] for result in lspline_results]

    # Create the plot
    plt.figure(figsize=(10, 6))


    # arch_colors = generate_colors(len(archs_pos))
    # # plot lines
    # for idx, (arch, params) in enumerate(archs_pos):
    #     # Plot a vertical line at each position in spline_params
    #     plt.axvline(x=params, label=arch, linestyle='-', color=arch_colors[idx])

    # Plot each model's results
    plt.plot(labels, bspline_values, marker='o', linestyle='-', color='blue', label="BSpline Model")
    plt.plot(labels, lspline_values, marker='o', linestyle='-', color='green', label="LSpline Model")
    plt.plot(labels, relu_values, marker='o', linestyle='-', color='black', label="ReLU Model")

    # Labeling the axes and the plot
    plt.xlabel("Number of Layers")
    plt.ylabel("Forward Latency")
    # plt.title("Architecture Comparison: Forward Latency ")
    plt.legend(loc="best")

    ax = plt.gca()
    ax.set_xticks(range(len(labels)))        # positions for ticks
    ax.set_xticklabels(map(lambda x: len(x), archs))   # blank out the default text

    # Show the plot
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def graph_params_x_flops(archs):

    input_dim = 8
    batch_size = 10
    
    relu_results = []
    bspline_results = []
    lspline_results = []

    
    for arch in archs:
        success = [False, False, False]
        while(False in success):
            try:
                relu_results.append((arch, profile_flops("ReLU", arch)[0]))
                success[0] = True
            except Exception:
                print("profile failed... trying again")
            
            try:
                bspline_results.append((arch, profile_flops("BSpline", arch)[0]))
                success[1] = True
            except Exception:
                print("profile failed... trying again")

            try:
                lspline_results.append((arch, profile_flops("LSpline", arch)[0]))
                success[2] = True
            except Exception:
                print("profile failed... trying again")
        clear_output()
        print(f"arch {arch} done")

    clear_output()

    labels = [str(result[0]) for result in relu_results]
    relu_values = [result[1] for result in relu_results]
    bspline_values = [result[1] for result in bspline_results]
    lspline_values = [result[1] for result in lspline_results]

    plt.figure(figsize=(10, 6))
    
    plt.plot(labels, bspline_values, marker='o', linestyle='-', color='blue', label="BSpline Model")
    plt.plot(labels, lspline_values, marker='o', linestyle='-', color='green', label="LSpline Model")
    plt.plot(labels, relu_values, marker='o', linestyle='-', color='black', label="ReLU Model")

    plt.xlabel("Number of Layers")
    plt.ylabel("FLOPs")
    plt.legend(loc="best")

    ax = plt.gca()
    ax.set_xticks(range(len(labels)))        # positions for ticks
    ax.set_xticklabels(map(lambda x: len(x), archs))   # blank out the default text

    # Show the plot
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_static_activation(sel, r=10):

    acts = {
        "relu": lambda x: np.maximum(0, x),
        "leaky_relu": lambda x: np.where(x > 0, x, 0.1 * x),
        "sig": lambda x: 1 / (1 + np.exp(-x)),
        "tanh": lambda x: (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)),
    }

    x = np.linspace(-1*r, r, 100)
    out = acts[sel](x)
    plt.plot(x, out, 'black', label='ReLU')
    plt.grid(True)

def profile_flops(arch, layers, ctrl=3, range_=1, input_size=8, batch_size=10):

    if(arch == "ReLU"):
        model = LinearReLU(layers)
    elif(arch == "BSpline"):
        model = LinearBSpline(layers, ctrl, range_)
    elif(arch == "LSpline"):
        model = linspline.LSplineFromBSpline(LinearBSpline(layers, ctrl, range_).get_layers())

    #^ FLOPs and Params

    base_flops, macs, params = get_model_profile(model=model, # model
            input_shape=(batch_size, input_size), # input shape to the model. If specified, the model takes a tensor with this shape as the only positional argument.
            args=None, # list of positional arguments to the model.
            kwargs=None, # dictionary of keyword arguments to the model.
            print_profile=False, #! prints the model graph with the measured profile attached to each module
            detailed=True, # print the detailed profile
            module_depth=-1, # depth into the nested modules, with -1 being the inner most modules
            top_modules=1, # the number of top modules to print aggregated profile
            warm_up=10, # the number of warm-ups before measuring the time of each module
            as_string=True, # print raw numbers (e.g. 1000) or as human-readable strings (e.g. 1k)
            output_file=None, # path to the output file. If None, the profiler prints to stdout.
            ignore_modules=None) # the list of modules to ignore in the profiling
    
    if(base_flops.__contains__("K")):
        flops = float(base_flops.replace("K", "").strip()) * 1000 / batch_size
    elif(base_flops.__contains__("M")):
        flops = float(base_flops.replace("M", "").strip()) * 1000000 / batch_size

    if(arch == "BSpline"):
        flops += utils.calc_bspline_flops(model) #! this might be incorrect- need to check w the implementation
    if(arch == "LSpline"):
        flops += utils.calc_lspline_flops(model)

    return flops, params

def profile_model(arch, layers, ctrl=3, range_=1, input_size=8, batch_size=10):

    if(arch == "ReLU"):
        model = LinearReLU(layers)
    elif(arch == "BSpline"):
        model = LinearBSpline(layers, ctrl, range_)
    elif(arch == "LSpline"):
        model = linspline.LSplineFromBSpline(LinearBSpline(layers, ctrl, range_).get_layers())

    #^ FLOPs and Params

    base_flops, macs, params = get_model_profile(model=model, # model
            input_shape=(batch_size, input_size), # input shape to the model. If specified, the model takes a tensor with this shape as the only positional argument.
            args=None, # list of positional arguments to the model.
            kwargs=None, # dictionary of keyword arguments to the model.
            print_profile=False, #! prints the model graph with the measured profile attached to each module
            detailed=True, # print the detailed profile
            module_depth=-1, # depth into the nested modules, with -1 being the inner most modules
            top_modules=1, # the number of top modules to print aggregated profile
            warm_up=10, # the number of warm-ups before measuring the time of each module
            as_string=True, # print raw numbers (e.g. 1000) or as human-readable strings (e.g. 1k)
            output_file=None, # path to the output file. If None, the profiler prints to stdout.
            ignore_modules=None) # the list of modules to ignore in the profiling
    
    if(base_flops.__contains__("K")):
        flops = float(base_flops.replace("K", "").strip()) * 1000 / batch_size
    elif(base_flops.__contains__("M")):
        flops = float(base_flops.replace("M", "").strip()) * 1000000 / batch_size
    else:
        flops = float(base_flops) / batch_size

    if(arch == "BSpline"):
        flops += utils.calc_bspline_flops(model) #! this might be incorrect- need to check w the implementation
    if(arch == "LSpline"):
        flops += utils.calc_lspline_flops(model)
    
    housing = fetch_california_housing()

    
    #^ Forward Latency

    per = 1000 # number of sims
    
    inpts = [ torch.rand(batch_size, input_size) * 3 - 1.5 for _ in range(per) ]
    start_time = time.perf_counter()
    for i in range(per):
        _ = model(inpts[i]) # model output is irrelevant
    end_time = time.perf_counter()
    fwd_lat_sim = round((end_time - start_time) * 1000 * 1000 / per / batch_size, 2) # per sample latency: seconds -> microsec per input

    
    X, y = torch.tensor(housing.data, dtype=torch.float32), torch.tensor(housing.target, dtype=torch.float32).reshape(-1, 1)
    # using a dataloader to randomize batching
    train_dataset = TensorDataset(X, y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    inpts = []
    for i in range(per):
        inpts.append(next(iter(train_loader))[0]) # pre-emptively do this so it doesn't affect timing
    
    start_time = time.perf_counter()
    for i in range(per):
        _ = model(inpts[i]) # model output is irrelevant
    end_time = time.perf_counter()
    
    fwd_lat_real = round((end_time - start_time) * 1000 * 1000 / per / batch_size, 2) # per sample latency: seconds -> microsec per input
    
    return flops, params, fwd_lat_real

def relu(x):
    return np.maximum(0, x)

def plot_bsplines(locs_list, coeffs_list, ncols=2, degree=1, scale_by_coeff=True, hide_bases=False):
    
    n = len(locs_list)

    # Determine the number of rows based on number of columns
    nrows = int(np.ceil(n / ncols))

    # Create the subplots
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, 
                             figsize=(6*ncols, 4.5*nrows), 
                             squeeze=False)  # squeeze=False to keep 2D array

    # Flatten axes for easy iteration
    axes_flat = axes.ravel()

    for i in range(n):
        locs = locs_list[i]
        coeffs = coeffs_list[i]
        ax = axes_flat[i]

        # Build the knot vector with repeated endpoints
        knots = np.concatenate((
            [locs[0]] * (degree + 1), 
            locs[1:-1], 
            [locs[-1]] * (degree + 1)
        ))
        
        # Construct the B-spline
        bspline = BSpline(knots, coeffs, degree)

        # Generate colors for control points
        colors = generate_colors(len(coeffs))

        # Plot ReLU on the same range for reference
        x_plot = np.linspace(min(locs), max(locs), 200)
        ax.plot(x_plot, relu(x_plot), 'grey', label='ReLU')

        # Plot the B-spline
        ax.plot(x_plot, bspline(x_plot), 'k', lw=2, alpha=0.7, label='BSpline')
        
        # Scatter the control points
        for j, (lx, ly) in enumerate(zip(locs, coeffs)):
            ax.scatter(lx, ly, color=colors[j], s=50, zorder=5)
            ax.text(lx, ly, f'$P_{{{j+1}}}$', color=colors[j], 
                    ha='right', va='bottom', fontsize=9)

        ax.set_title(f'B-Spline #{i+1}')
        ax.legend()
        ax.grid(True)

    # Hide any unused subplots if locs_list and coeffs_list are not square
    for i in range(n, nrows*ncols):
        axes_flat[i].axis('off')

    plt.tight_layout()
    plt.show()
    # return fig, axes

    import torch

def train_model_plot_layer_activations(layers=[8], epochs={"relu": 0, "both": 0, "bspline": 0}, ncols=4):

    # Load the data
    housing = fetch_california_housing()
    X, y = housing.data, housing.target

    # train-test split of the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.85, shuffle=True)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

    # using a dataloader to randomize batching
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)



    bsm = LinearBSpline(layers, 3, 1, "relu")
    optimizer = optim.Adam(bsm.parameters_no_deepspline(), lr=0.001)
    aux_optimizer = optim.Adam(bsm.parameters_deepspline(), lr=0.0001)
    loss_fn = nn.MSELoss()  # mean square error
    lmbda = 1e-4 # regularization weight

    print("Init complete")

    bsm.train()
    for mode in ["relu", "both", "bspline"]:
        print(f"Training {mode}")
        for epoch in range(epochs[mode]):
            rloss = 0
            for X_batch, y_batch in train_loader:
                if mode in ["relu", "both"]:
                    optimizer.zero_grad()
                if mode in ["bspline", "both"]:
                    aux_optimizer.zero_grad()

                y_pred = bsm(X_batch)
                loss = loss_fn(y_pred, y_batch)
                loss2 = loss + lmbda * bsm.TV2()
                loss2.backward()
                if mode in ["relu", "both"]:
                    optimizer.step()
                if mode in ["bspline", "both"]:
                    aux_optimizer.step()
                rloss += loss
            if(epoch % 5 == 0):
                print(f"Epoch {epoch}: {rloss}")
    
    print("All training complete")

    layer_locs = []
    layer_coeffs = []

    zeroed_count = 0

    for layer in bsm.get_layers():
        if(type(layer) is deepsplines.ds_modules.deepBspline.DeepBSpline):
            coeffs = layer.coefficients_vect.view(layer.num_activations, layer.size).detach()
            layer_locs.append(layer.grid_tensor.detach())
            layer_coeffs.append(coeffs)
            
            if(torch.sum(torch.abs(coeffs)) < .1): # then we say its zeroed
                zeroed_count += 1

    layer_to_plot = 0
    utils.plot_bsplines(layer_locs[layer_to_plot], layer_coeffs[layer_to_plot], ncols)

def train_models_count_zeroed_AFs(
        save_output = "zeroed_info",
        runs = 1, 
        layers=[8], 
        epochs={"relu": 200, "both": 0, "bspline": 10},
        zero_thres = 0.05
    ):

    # Load the data
    housing = fetch_california_housing()
    X, y = housing.data, housing.target

    losses = []
    zeroed_counter = []

    for r in range(runs): 

        losses.append([])

        # train-test split of the dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.85, shuffle=True)

        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)

        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

        # using a dataloader to randomize batching
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)

        bsm = LinearBSpline(layers, 3, 1, "relu")
        optimizer = optim.Adam(bsm.parameters_no_deepspline(), lr=0.001)
        aux_optimizer = optim.Adam(bsm.parameters_deepspline(), lr=0.0001)
        loss_fn = nn.MSELoss()  # mean square error
        lmbda = 1e-4 # regularization weight

        bsm.train()
        for mode in ["relu", "both", "bspline"]:
            for _ in range(epochs[mode]):
                rloss = 0
                for X_batch, y_batch in train_loader:
                    if mode in ["relu", "both"]:
                        optimizer.zero_grad()
                    if mode in ["bspline", "both"]:
                        aux_optimizer.zero_grad()

                    y_pred = bsm(X_batch)
                    loss = loss_fn(y_pred, y_batch)
                    loss2 = loss + lmbda * bsm.TV2()
                    loss2.backward()
                    if mode in ["relu", "both"]:
                        optimizer.step()
                    if mode in ["bspline", "both"]:
                        aux_optimizer.step()
                    rloss += loss
                losses[r].append(loss.item())

        layer_locs = []
        layer_coeffs = []
        zeroed_count = 0

        for idx,layer in enumerate(bsm.get_layers()):
            zeroed_counter.append([]) # count on a per layer basis
            if(type(layer) is deepsplines.ds_modules.deepBspline.DeepBSpline):
                coeffs = layer.coefficients_vect.view(layer.num_activations, layer.size).detach()
                layer_locs.append(layer.grid_tensor.detach())
                layer_coeffs.append(coeffs)
                
                for spline_idx in range(len(coeffs)):
                    score = torch.sum(torch.abs(coeffs[spline_idx]))
                    if(score < .05): # then we say its zeroed
                        zeroed_count += 1
        
                zeroed_counter[idx].append(zeroed_count)

    data = {
        "zero_thres": zero_thres,
        "zeroed_counter": zeroed_counter,
        "losses": losses
    }

    with open(f'saved/{save_output}.json', 'w') as f:
        json.dump(data, f)

def test_cycle_training(layers=[8], wb_epochs=[100,50,50,50], af_epochs=[10,10,10,10], y_clamp=None):

    # Load the data
    housing = fetch_california_housing()
    X, y = housing.data, housing.target

    # train-test split of the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.85, shuffle=True)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

    # using a dataloader to randomize batching
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)



    bsm = LinearBSpline(layers, 3, 1, "relu")
    optimizer = optim.Adam(bsm.parameters_no_deepspline(), lr=0.001)
    aux_optimizer = optim.Adam(bsm.parameters_deepspline(), lr=0.0001)
    loss_fn = nn.MSELoss()  # mean square error
    lmbda = 1e-4 # regularization weight

    print("Init complete")

    save_loss = []

    bsm.train()
    for cycle in range(len(wb_epochs)):
        print(f"Training relu {wb_epochs[cycle]}")
        for _ in range(wb_epochs[cycle]):
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                y_pred = bsm(X_batch)
                loss = loss_fn(y_pred, y_batch)
                loss2 = loss + lmbda * bsm.TV2()
                loss2.backward()
                optimizer.step()
            save_loss.append(loss.detach().item())
        print(f"Loss at end of ReLU cycle {cycle},{wb_epochs[cycle]}: {loss}")

        print(f"Training bspline {af_epochs[cycle]}")
        for _ in range(af_epochs[cycle]):
            for X_batch, y_batch in train_loader:
                aux_optimizer.zero_grad()
                y_pred = bsm(X_batch)
                loss = loss_fn(y_pred, y_batch)
                loss2 = loss + lmbda * bsm.TV2()
                loss2.backward()
                aux_optimizer.step()
            save_loss.append(loss.detach().item())
        print(f"Loss at end of Bspline cycle {cycle},{af_epochs[cycle]}: {loss}")
    
    print("Training complete")

    data = {
        "Epoch": range(len(save_loss)),
        "Val Loss": save_loss
    }
    df = pd.DataFrame(data)

    return df

def CPU_vs_GPU(layers=[8]):
    return

def get_model_zeroed_activations(model):
    zeroed = []
    if(type(model) is LinearBSpline):
        bspline_layer_idx = 0
        for layer in model.get_layers():
            if(type(layer) is deepsplines.ds_modules.deepBspline.DeepBSpline):
                zeroed.append([]) # layer level within run
                coeffs = layer.coefficients_vect.view(layer.num_activations, layer.size).detach()
                
                zeroed[bspline_layer_idx] = 0
                for spline_idx in range(len(coeffs)):
                    score = torch.sum(torch.abs(coeffs[spline_idx]))
                    if(score < .05): # then we say its zeroed
                        zeroed[bspline_layer_idx] += 1

                bspline_layer_idx += 1
    return zeroed
