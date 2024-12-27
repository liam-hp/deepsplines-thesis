import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from datetime import datetime
import time

from omegaconf import DictConfig, OmegaConf, ListConfig
import hydra

from models import LinearReLU, LinearBSpline
from deepspeed.profiling.flops_profiler import get_model_profile

from utils import display_vals_table, time_str_to_timedelta, timedelta_to_time_str, calc_bspline_flops
import json
import numpy as np
from datetime import datetime, timedelta
from torch.utils.data import TensorDataset, DataLoader

class Config:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self):
        def convert(value):
            if isinstance(value, ListConfig):
                print(value)
                return OmegaConf.to_object(value)
            return value

        return {key: convert(value) for key, value in self.__dict__.items()}


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
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.85, shuffle=True)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

    # using a dataloader to randomize batching
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=tparams.batch_size, shuffle=True)
    
    loss_fn = nn.MSELoss()  # mean square error

    train_history = []
    val_history = []
    train_times = []

    start_time = datetime.now()

    for arch in ["relu", "bspline", "both"]:
        
        # set specifications based on the architecture
        if(arch == "relu"):
            if(tparams.relu_epochs == 0):
                continue
            model = LinearReLU(mparams.layers)
            epochs = tparams.relu_epochs

            optimizer = optim.Adam(model.parameters(), lr=tparams.lr_wb)
            if(tparams.lrs == "steplr"):
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=tparams.lrs_stepsize, gamma=tparams.lrs_gamma)

        elif(arch == "bspline" or arch == "both"):
            if(arch == "bspline"):
                if(tparams.spline_epochs == 0):
                    continue
                if(tparams.relu_epochs > 0): # if we pretrained on ReLU, load those weights
                    torch.save(model.state_dict(), "./saved_model.pt")
                    model = LinearBSpline(mparams.layers, mparams.cpoints, mparams.range_)
                    model.load_state_dict(torch.load("./saved_model.pt", weights_only=True), strict=False)
                else:
                    model = LinearBSpline(mparams.layers, mparams.cpoints, mparams.range_)

            if(arch == "both"):
                if(tparams.both_epochs == 0):
                    continue

                if(tparams.spline_epochs == 0):
                    if(tparams.relu_epochs > 0): # if we pretrained on ReLU, load those weights
                        torch.save(model.state_dict(), "./saved_model.pt")
                        model = LinearBSpline(mparams.layers, mparams.cpoints, mparams.range_)
                        model.load_state_dict(torch.load("./saved_model.pt", weights_only=True), strict=False)
                    else:
                        model = LinearBSpline(mparams.layers, mparams.cpoints, mparams.range_)

            epochs = tparams.spline_epochs
            optimizer = optim.Adam(model.parameters_no_deepspline(), lr=tparams.lr_wb)
            aux_optimizer = optim.Adam(model.parameters_deepspline(), lr=tparams.lr_bs)

            if(tparams.lrs == "steplr"):
                # resetting scheduler
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=tparams.lrs_stepsize, gamma=tparams.lrs_gamma)
                aux_scheduler = torch.optim.lr_scheduler.StepLR(aux_optimizer, step_size=tparams.lrs_stepsize, gamma=tparams.lrs_gamma)
            
            lmbda = 1e-4 # regularization weight
            lipschitz = False # lipschitz control
        
        # training loop
        for _ in range(epochs):

            # train the model
            model.train()
            epoch_loss = 0
            epoch_start = datetime.now()

            for X_batch, y_batch in train_loader:

                # forward pass + get loss
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                if(arch == "relu" or arch == "both"):
                    optimizer.zero_grad()
                if(arch == "bspline" or arch == "both"):
                    aux_optimizer.zero_grad()
                    if lipschitz is True:
                        loss = loss + lmbda * model.BV2()
                    else:
                        loss = loss + lmbda * model.TV2()
                epoch_loss += float(loss) * len(X_batch)

                # compute gradient and step on the optimizer
                loss.backward()
                if(arch == "relu" or arch == "both"):
                    optimizer.step()
                if(arch == "bspline" or arch == "both"):
                    aux_optimizer.step()                
            
            # step the LR scheduler
            if(tparams.lrs != "None"):
                if((arch == "bspline" or arch == "both") and scheduler.get_last_lr()[0] * tparams.lrs_gamma > 0.0001):
                    scheduler.step()

                if((arch=="bspline" or arch=="both") and aux_scheduler.get_last_lr()[0] * tparams.lrs_gamma > 0.00001):
                    aux_scheduler.step()

            # normalize to per-input MSE 
            avg_epoch_loss = epoch_loss / len(X_train)
            train_history.append(avg_epoch_loss)

            train_times.append(datetime.now() - epoch_start)

            # run on validation data
            model.eval()
            y_pred = model(X_test) # pass in all the validation data
            loss = float(loss_fn(y_pred, y_test))
            val_history.append(round(loss, 4))

    training_time = datetime.now() - start_time

    # compute forward latency
    start_time = time.perf_counter()
    _ = model(X_test) # model output is irrelevant
    end_time = time.perf_counter()
    fwd_lat = (end_time - start_time) / len(X_test) * 1000 # per sample latency: seconds -> milliseconds
    fwd_lat = round(fwd_lat, 4)

    final_locs = None
    final_coeffs = None
    if(tparams.spline_epochs > 0):
        final_locs = model.get_deepspline_activations()[0]['locations']
        final_coeffs = model.get_deepspline_activations()[0]['coefficients']

    return train_history, val_history, train_times, training_time, fwd_lat, model, final_locs, final_coeffs


@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg: DictConfig) -> None:

    if(cfg.runs == 0):
        return
    
    print("Parameters: ")
    print(OmegaConf.to_yaml(cfg))

    # By default, PyTorch attempts to use all available CPU cores for intra-op parallelism. Set threads = cpu cores
    torch.set_num_threads(cfg.threads)

    # Load the data
    housing = fetch_california_housing()
    X, y = housing.data, housing.target

    run_summaries = []
    run_validations = [] # validations
    run_history = [] # training 
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
        both_epochs = cfg.both_epochs,
        batch_size = cfg.batch_size,
        lr_wb = cfg.lr_wb,
        lr_bs = cfg.lr_bs,
        lrs = cfg.lrs,
        lrs_gamma = cfg.lrs_gamma,
        lrs_stepsize = cfg.lrs_stepsize
    )
    
    for r in range(cfg.runs):
        train_history, val_history, epoch_times, training_time, fwd_lat, model, _, _ = training_run(mparams, tparams, X, y)
        run_summaries.append([r+1, min(val_history), val_history[-1], f"{training_time}", fwd_lat ])
        run_validations.append(val_history)
        run_history.append(train_history)
        run_epoch_times.append([t.microseconds / 1000 for t in epoch_times])
        print(f"Run {r} complete: {[min(val_history), val_history[-1], f'{training_time}']}")

    avg = sum(list(map(lambda r: r[2], run_summaries)))/len(run_summaries)
    for run in run_summaries:
        if(run[2] > avg*1.5):
            outliers += 1
            run.append(True) # replace w how close it is to avg
        else:
            run.append(False)
    
    rerun = 0
    if(outliers > 0):
        print(f"rerunning {outliers} outliers...")

    while(rerun < outliers):
        train_history, val_history, epoch_times, training_time, fwd_lat, model, _, _ = training_run(mparams, tparams, X, y)

        if(val_history[-1] <= avg*1.5):
            run_summaries.append([f"rerun {rerun}", min(val_history), val_history[-1], f"{training_time}", fwd_lat, False ])
            run_validations.append(val_history)
            run_history.append(train_history)
            run_epoch_times.append([t.microseconds / 1000 for t in epoch_times])

            print(f"Rerun {rerun} complete: {[min(val_history), val_history[-1], f'{training_time}']}")
            rerun += 1 
    
    print(f"\nOutliers: {outliers} \n")

    display_vals_table(run_summaries.copy())
    
    try:
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
    except:
        print("get_model_profile failed... skipping")
        base_flops = "-1000K"
        params = -1000000

    flops = float(base_flops.replace("K", "").strip()) * 1000 / cfg.batch_size
    if(cfg.spline_epochs > 0 or cfg.both_epochs > 0):
        flops += calc_bspline_flops(model)

    print(f"\n{flops} FLOPs, {params} Params \n")

    time_deltas = [time_str_to_timedelta(row[3]) for row in run_summaries]
    fwd_latency_vals = [row[4] for row in run_summaries]
    average_time_delta = sum(time_deltas, timedelta()) / len(time_deltas)
    
    average_time = timedelta_to_time_str(average_time_delta)
    average_fwd_latency = round(np.mean(fwd_latency_vals), 4)

    data = {
        "mparams": mparams.to_dict(),
        "tparams": tparams.to_dict(),
        "flops": flops,
        "params": params,
        "avg_time": average_time,
        "avg_fwdlat": average_fwd_latency,
        "times": run_epoch_times,
        "vals": run_validations,
        "trains": run_history,
    }

    if(tparams.lrs == "steplr"):
        out = f"{cfg.layers}_({cfg.lr_wb},{cfg.lr_bs},{cfg.lrs_gamma},{cfg.lrs_stepsize})_({cfg.relu_epochs},{cfg.spline_epochs},{cfg.both_epochs})_{cfg.runs}"
    else:
        out = f"{cfg.layers}_({cfg.lr_wb},{cfg.lr_bs})_({cfg.relu_epochs},{cfg.spline_epochs},{cfg.both_epochs})_{cfg.runs}"

    if(cfg.add_to_out != ""):
        out += "_" + cfg.add_to_out

    with open(f'saved/{out}.json', 'w') as f:
        json.dump(data, f)
    
    print(f"Saved to saved/{out}")

    return

if __name__ == "__main__":
    my_app()