import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from datetime import datetime
import time

from omegaconf import DictConfig, OmegaConf
import hydra

from models import LinearReLU, LinearBSpline
from deepspeed.profiling.flops_profiler import get_model_profile

from utils import display_vals_table


class Config:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def training_run(mparams, tparams, X, y):
    ''' 
        Initializes and trains a specific architecture once

        Params:
            mparams: 
                layers: int[], describes the inner layers of the network you want to initialize.
                    example: [24,8] will initialize the network as 8 (auto, input dim) -> 24 -> 8 -> 1 (auto, output dim)
                cpoints: int, number of control points for the bspline
                range_: int, range of the bspline

            tparams: 
                relu_epochs: int, number of epochs to train on relu
                spline_epochs: int, number of epochs to train on bspline
                batch_size: int, batch size
                lr: float, learning rate

            X: float[][], input data
            y: float[], output data

        Returns:
            train_history: float[], training loss at each epoch
            val_history: float[], validation loss at each epoch
            epoch_times: int[], clock-time at each epoch
            training_time: datetime, total training time, should be same as epoch_times[-1]
            fwd_lat: milliseconds it takes to process 1 input (check this again)
            model: return the model itself

    '''

    # train-test split of the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

    loss_fn = nn.MSELoss()  # mean square error

    train_history = []
    val_history = []
    epoch_times = []

    start_time = datetime.now()

    for arch in ["relu", "bspline"]:
        
        # set specifications based on the architecture
        if(arch == "relu"):
            if(tparams.relu_epochs == 0):
                continue
            model = LinearReLU(mparams.layers)
            epochs = tparams.relu_epochs

            optimizer = optim.Adam(model.parameters(), lr=tparams.lr_wb)

        elif(arch == "bspline"):
            if(tparams.spline_epochs == 0):
                continue

            if(tparams.relu_epochs > 0): # if we pretrained on ReLU, load those weights
                torch.save(model.state_dict(), "./saved_model.pt")
                model = LinearBSpline(mparams.layers, mparams.cpoints, mparams.range_)
                model.load_state_dict(torch.load("./saved_model.pt", weights_only=True), strict=False)
            else:
                model = LinearBSpline(mparams.layers, mparams.cpoints, mparams.range_)

            epochs = tparams.spline_epochs
            optimizer = optim.Adam(model.parameters_no_deepspline(), lr=tparams.lr_wb)
            aux_optimizer = optim.Adam(model.parameters_deepspline(), lr=tparams.lr_bs)
            lmbda = 1e-4 # regularization weight
            lipschitz = False # lipschitz control

        for _ in range(epochs):
            model.train()
            epoch_loss = 0
            for start in range(0, len(X_train), tparams.batch_size):
                # select batch
                X_batch = X_train[start:start + tparams.batch_size]
                y_batch = y_train[start:start + tparams.batch_size]

                # forward pass
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)

                # backward pass
                optimizer.zero_grad()
                if(arch == "bspline"):
                    aux_optimizer.zero_grad()

                    # add regularization loss
                    if lipschitz is True:
                        loss = loss + lmbda * model.BV2()
                    else:
                        loss = loss + lmbda * model.TV2()

                # gradient descent
                loss.backward()

                # update weights
                optimizer.step()
                if(arch=="bspline"):
                    aux_optimizer.step()

                epoch_loss += float(loss) * len(X_batch)

            avg_epoch_loss = epoch_loss / len(X_train)
            train_history.append(avg_epoch_loss)

            # run on validation data
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
    fwd_lat = ((end_time - start_time) * 1000) / len(X_test) * 1000 # milliseconds (?) - per sample latency
    fwd_lat = round(fwd_lat, 4)

    final_locs = None
    final_coeffs = None
    if(tparams.spline_epochs > 0):
        final_locs = model.get_deepspline_activations()[0]['locations']
        final_coeffs = model.get_deepspline_activations()[0]['coefficients']

    return train_history, val_history, epoch_times, training_time, fwd_lat, model, final_locs, final_coeffs


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
    
    mparams = Config(
        layers = cfg.layers,
        cpoints = cfg.cpoints,
        range_ = cfg.range,
    )
            
    tparams = Config(
        relu_epochs = cfg.relu_epochs,
        spline_epochs = cfg.spline_epochs,
        batch_size = cfg.batch_size,
        lr_wb = cfg.lr_wb,
        lr_bs = cfg.lr_bs
    )
    
    for r in range(cfg.runs):
        train_history, val_history, epoch_times, training_time, fwd_lat, model, _, _ = training_run(mparams, tparams, X, y)
        run_summaries.append([r+1, min(val_history), val_history[-1], f"{training_time}", fwd_lat ])
        run_validations.append(val_history)
        run_epoch_times.append([t.seconds for t in epoch_times])
        print(f"Run {r} complete: {[min(val_history), val_history[-1], f"{training_time}"]}")

    avg = sum(list(map(lambda r: r[1], run_summaries)))
    for r in run_summaries:
        if(r[1] > avg*1.5):
            outliers += 1
            r.append(False)
        else:
            r.append(True)
    
    reruns = outliers
    while(reruns > 0):
        train_history, val_history, epoch_times, training_time, fwd_lat, model = training_run(mparams, tparams, X, y)
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
    
    base_flops, macs, params = get_model_profile(model=model, # model
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
    
    
    fwd_spline_flops = 0

    # MLP sums before activation
    d_in = 1
    d_out = 1
    
    K = 1 # degree of the spline

    flops = float(base_flops.replace("K", "").strip()) * 1000 / cfg.batch_size
    if(cfg.spline_epochs > 0):
        for act in model.get_deepspline_activations(): # for each layer
            num_activations = len(act['locations']) # the number of bsplines on that layer
            G = len(act['locations'][0]) - 1 # number of control points - 1
            fwd_spline_flops += (d_in * d_out * (9*K * (G + 1.5 * K) + 2 * G - 2.5 * K + 3)) * num_activations
            # ! Do we multiply by num_activations?? unclear from the fairer paper..

        flops += fwd_spline_flops # flops per input

    print(f"\n{flops} FLOPs, {params} Params \n")

    print("\n\n\n Run times: \n")
    print(run_epoch_times)
    print("\n\n\n Run vals: \n")
    print(run_validations)

if __name__ == "__main__":
    my_app()