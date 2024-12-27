import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_california_housing
import torch
from sklearn.model_selection import train_test_split
from datetime import datetime
import time
from deepspeed.profiling.flops_profiler import get_model_profile
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig, OmegaConf

from models import LinearReLU, LinearBSpline
from utils import display_vals_table, display_vals_plot

def get_model(arch, r=True):
    which, layers, sparam = arch.split("_")
    layers = list(map(lambda s: int(s), layers.split("-")))

    match(which):
        case "ReLU":
            return LinearReLU(layers)
        case "BSpline":
            sparam = list(map(lambda s: int(s), sparam.split("-")))
            return LinearBSpline(layers, sparam[0], sparam[1], 'relu')
        case "Transpl":
            if(r):
                return LinearReLU(layers)
            else:
                sparam = list(map(lambda s: int(s), sparam.split("-")))
                return LinearBSpline(layers, sparam[0], sparam[1], 'relu')

def training_run(cfg, X, y):
    # train-test split of the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

    model = get_model(cfg.arch)
    loss_fn = nn.MSELoss()  # mean square error
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    if(cfg.arch[0]=='B'):
        optimizer = optim.Adam(model.parameters_no_deepspline())
        aux_optimizer = optim.Adam(model.parameters_deepspline())
        lmbda = 1e-4 # regularization weight
        lipschitz = False # lipschitz control

    train_history = []
    val_history = []
    epoch_times = []

    start_time = datetime.now()

    for _ in range(cfg.n_epochs):
        model.train()
        epoch_loss = 0
        for start in range(0, len(X_train), cfg.batch_size):
            # take a batch
            X_batch = X_train[start:start + cfg.batch_size]
            y_batch = y_train[start:start + cfg.batch_size]
            # forward pass
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            # backward pass
            optimizer.zero_grad()

            if(cfg.arch[0]=='B'):
                aux_optimizer.zero_grad()
                # add regularization loss
                if lipschitz is True:
                    loss = loss + lmbda * model.BV2()
                else:
                    loss = loss + lmbda * model.TV2()

            loss.backward()

            # update weights
            optimizer.step()
            if(cfg.arch[0]=='B'):
                aux_optimizer.step()

            epoch_loss += float(loss) * len(X_batch)
        avg_epoch_loss = epoch_loss / len(X_train)
        train_history.append(avg_epoch_loss)

        # evaluate accuracy at end of each epoch
        model.eval()
        y_pred = model(X_test)
        mse = float(loss_fn(y_pred, y_test))

        val_history.append(round(mse, 4))
        epoch_times.append(datetime.now() - start_time)

    training_time = datetime.now() - start_time

    # computing fwd latency
    start_time = time.perf_counter()
    _ = model(X_test)
    end_time = time.perf_counter()
    fwd_lat = ((end_time - start_time) * 1000) / len(X_test) * 1000 # microseconds- per sample latency
    fwd_lat = round(fwd_lat, 4)

    return train_history, val_history, epoch_times, training_time, fwd_lat, model

def training_run_2(cfg, X, y):
    # train-test split of the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

    start_time = datetime.now()

    for stage in ["pretraining", "withsplines"]:
        if(stage == "pretraining"):
            model = get_model(cfg.arch)
            optimizer = optim.Adam(model.parameters(), lr=0.0001)
            n_epochs = cfg.relu_epochs
            train_history = []
            val_history = []
            epoch_times = []

        elif(stage == "withsplines"):
            model = get_model(cfg.arch, False)
            model.load_state_dict(torch.load("./saved_model.pt", weights_only=True), strict=False)
            optimizer = optim.Adam(model.parameters_no_deepspline())
            aux_optimizer = optim.Adam(model.parameters_deepspline())
            lmbda = 1e-4 # regularization weight
            lipschitz = False # lipschitz control

            n_epochs = cfg.spline_epochs

        loss_fn = nn.MSELoss()  # mean square error

        for _ in range(n_epochs):
            model.train()
            epoch_loss = 0
            for start in range(0, len(X_train), cfg.batch_size):
                # take a batch
                X_batch = X_train[start:start + cfg.batch_size]
                y_batch = y_train[start:start + cfg.batch_size]
                # forward pass
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                # backward pass
                optimizer.zero_grad()

                if(stage == "withsplines"):
                    aux_optimizer.zero_grad()
                    # add regularization loss
                    if lipschitz is True:
                        loss = loss + lmbda * model.BV2()
                    else:
                        loss = loss + lmbda * model.TV2()

                loss.backward()

                # update weights
                optimizer.step()

                if(stage == "withsplines"):
                    aux_optimizer.step()

                epoch_loss += float(loss) * len(X_batch)
            avg_epoch_loss = epoch_loss / len(X_train)
            train_history.append(avg_epoch_loss)

            # evaluate accuracy at end of each epoch
            model.eval()
            y_pred = model(X_test)
            mse = float(loss_fn(y_pred, y_test))

            val_history.append(round(mse, 4))
            epoch_times.append(datetime.now() - start_time)

        if(stage == "pretraining"):
            torch.save(model.state_dict(), "./saved_model.pt")

    training_time = datetime.now() - start_time
    start_time = time.perf_counter()
    _ = model(X_test)
    end_time = time.perf_counter()
    fwd_lat = ((end_time - start_time) * 1000) / len(X_test) * 1000 # microseconds- per sample latency
    fwd_lat = round(fwd_lat, 4)
    
    return train_history, val_history, epoch_times, training_time, fwd_lat, model

@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg: DictConfig) -> None:
    print("Parameters: ")
    print(OmegaConf.to_yaml(cfg))

    # By default, PyTorch attempts to use all available CPU cores for intra-op parallelism. Set threads = cpu cores
    torch.set_num_threads(cfg.threads)

    # Load the data
    housing = fetch_california_housing()
    X, y = housing.data, housing.target

    run_summaries = []
    run_validations = []
    run_epoch_times = []
    outliers = 0

    if(cfg.arch[0] == 'R' or cfg.arch[0] == 'B'): # either pure ReLU or pure BSpline
        print("Training ReLU or BSpline")
        for r in range(cfg.runs):
            
            train_history, val_history, epoch_times, training_time, fwd_lat, model = training_run(cfg, X, y)

            run_summaries.append([r+1, min(val_history), val_history[-1], f"{training_time}", fwd_lat ])
            run_validations.append(val_history)
            run_epoch_times.append([t.seconds for t in epoch_times])
            print(f"Run {r} complete.")

        avg = sum(list(map(lambda r: r[1], run_summaries)))
        for r in run_summaries:
            if(r[1] > avg*1.5):
                outliers += 1
                r.append(False)
            else:
                r.append(True)
        
        reruns = outliers
        while(reruns > 0):
            train_history, val_history, epoch_times, training_time, fwd_lat, model = training_run(cfg, X, y)
            run_summaries.append([r+1, min(val_history), val_history[-1], f"{training_time}", fwd_lat ])
            run_validations.append(val_history)
            run_epoch_times.append([t.seconds for t in epoch_times])
            print(f"Run {r} complete.")

            if(min(val_history) > avg*1.5):
                outliers += 1
                reruns += 1
                run_summaries[-1].append(False)
            else:
                reruns -= 1
                run_summaries[-1].append(True)  

    elif(cfg.arch[0] == 'T'): # transplant
        print("Training ReLU -> BSpline")
        for r in range(cfg.runs):
            
            train_history, val_history, epoch_times, training_time, fwd_lat, model = training_run_2(cfg, X, y)

            run_summaries.append([r+1, min(val_history), val_history[-1], f"{training_time}", fwd_lat ])
            run_validations.append(val_history)
            run_epoch_times.append([t.seconds for t in epoch_times])
            print(f"Run {r} complete.")

            avg = sum(list(map(lambda r: r[1], run_summaries)))
        for r in run_summaries:
            if(r[1] > avg*1.5):
                outliers += 1
                r.append(False)
            else:
                r.append(True)
        
        reruns = outliers
        while(reruns > 0):
            train_history, val_history, epoch_times, training_time, fwd_lat, model = training_run_2(cfg, X, y)
            run_summaries.append([r+1, min(val_history), val_history[-1], f"{training_time}", fwd_lat ])
            run_validations.append(val_history)
            run_epoch_times.append([t.seconds for t in epoch_times])
            print(f"Run {r} complete.")

            if(min(val_history) > avg*1.5):
                outliers += 1
                reruns += 1
                run_summaries[-1].append(False)
            else:
                reruns -= 1
                run_summaries[-1].append(True)
        
    print(f"\nOutliers: {outliers} \n")
    display_vals_table(run_summaries)

    flops, macs, params = get_model_profile(model=model, # model
                                    input_shape=(cfg.batch_size, 8), # input shape to the model. If specified, the model takes a tensor with this shape as the only positional argument.
                                    args=None, # list of positional arguments to the model.
                                    kwargs=None, # dictionary of keyword arguments to the model.
                                    print_profile=False, # prints the model graph with the measured profile attached to each module
                                    detailed=True, # print the detailed profile
                                    module_depth=-1, # depth into the nested modules, with -1 being the inner most modules
                                    top_modules=1, # the number of top modules to print aggregated profile
                                    warm_up=10, # the number of warm-ups before measuring the time of each module
                                    as_string=True, # print raw numbers (e.g. 1000) or as human-readable strings (e.g. 1k)
                                    output_file=None, # path to the output file. If None, the profiler prints to stdout.
                                    ignore_modules=None) # the list of modules to ignore in the profiling
    
    
    if(cfg['arch'][0] == 'B' or cfg['arch'][0] == 'T'):
        fwd_spline_flops = 0
        d_in = 1 # MLP sums before activation
        d_out = 1 # 

        K = 1 # degree of the spline

        for act in model.get_deepspline_activations(): # for each layer
            num_activations = len(act['locations']) # the number of bsplines on that layer
            G = len(act['locations'][0]) - 1 # number of control points - 1
            fwd_spline_flops += (d_in * d_out * (9*K * (G + 1.5 * K) + 2 * G - 2.5 * K + 3)) * num_activations
            # ! Do we multiply by num_activations?? unclear from the fairer paper..
            
        fwd_spline_flops *= cfg.batch_size # multiply by batch size

        print(fwd_spline_flops)
        print(f'{float(flops.replace("K", "").strip()) * 1000 + fwd_spline_flops}')

    print(f"\n{cfg.arch}: {flops} FLOPs, {params} Params on batch size {cfg.batch_size} \n")

    print("\n Run times: \n")
    print(run_epoch_times)
    print("\n Run vals: \n")
    print(run_validations)

if __name__ == "__main__":
    my_app()