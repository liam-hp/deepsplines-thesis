from tabulate import tabulate
from datetime import datetime, timedelta
import numpy as np
import math, json
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
from itertools import accumulate
import linspline

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

def generate_colors(n):
      cmap = plt.cm.rainbow
      # Generate n evenly spaced values between 0 and 1
      return cmap(np.linspace(0, 1, n))

def plot_bspline(locs, coeffs, degree=1, scale_by_coeff=True, hide_bases=False):
    
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
    ax1.legend()

    ax1.plot(x, bspline(x), 'black', lw=2, alpha=0.7, label='BSpline')
    for i, (x, y) in enumerate(zip(locs, coeffs)):
        ax1.scatter(x, y, color=colors[i], s=50, zorder=5, label='Control Points')
        ax1.text(x, y, f'$P_{{{i + 1}}}$', color=colors[i], ha='right', va='bottom')
    ax1.grid(True)
    ax1.set_title('B-Spline and Control Points')

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
                ax2.set_title('B-Spline Basis Functions (Not Scaled)')
        
        ax2.grid(True)

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

  # MLP sums before activation
  d_in = 1 
  d_out = 1

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
        if(layer is linspline.LinearSplineLayer):
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

def plot_multiple2(model_paths, x="time", xlim=None, ylim=None, xmin=None, ymin=None, vbars=[]):
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
    
    ax1.legend()

    ax2.axis('off')
    tbl = ax2.table(
        cellText=table_data,
        colLabels=["Final Loss"],
        loc='center'
    )
    tbl.scale(1.5, 1.5)

    for i, row in enumerate(table_data):
        tbl[(i+1, 0)]._text.set_color(colors[i])

    for (r, c), cell in tbl.get_celld().items():
        cell.set_text_props(ha='center', va='center')

    