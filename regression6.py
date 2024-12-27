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

from utils import calc_bspline_flops
import json
import numpy as np
from datetime import datetime, timedelta
from torch.utils.data import TensorDataset, DataLoader

import string, random, os

class Config:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self):
        def convert(value):
            if isinstance(value, ListConfig):
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

    # comp relu
    cr_val_history = []
    cr_train_history = []
    cr_train_times = []
    cr_fwd_lat = -1

    model_save_code = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
    
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
                    model = LinearBSpline(mparams.layers, mparams.cpoints, mparams.range_)
                    model.load_state_dict(torch.load(f"./temp_models/{model_save_code}.pt", weights_only=True), strict=False)
                else:
                    model = LinearBSpline(mparams.layers, mparams.cpoints, mparams.range_)
                epochs = tparams.spline_epochs

            if(arch == "both"):
                if(tparams.both_epochs == 0):
                    continue
                elif(tparams.spline_epochs == 0):
                    if(tparams.relu_epochs > 0): # if we pretrained on ReLU, load those weights
                        model = LinearBSpline(mparams.layers, mparams.cpoints, mparams.range_)
                        model.load_state_dict(torch.load(f"./temp_models/{model_save_code}.pt", weights_only=True), strict=False)
                    else:
                        model = LinearBSpline(mparams.layers, mparams.cpoints, mparams.range_)
                # else we just continue training on the same model
                epochs = tparams.both_epochs

            
            optimizer = optim.Adam(model.parameters_no_deepspline(), lr=tparams.lr_wb)
            aux_optimizer = optim.Adam(model.parameters_deepspline(), lr=tparams.lr_bs)

            if(tparams.lrs == "steplr"):
                # resetting scheduler
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=tparams.lrs_stepsize, gamma=tparams.lrs_gamma)
                aux_scheduler = torch.optim.lr_scheduler.StepLR(aux_optimizer, step_size=tparams.lrs_stepsize, gamma=tparams.lrs_gamma)
            
            lmbda = 1e-4 # regularization weight
            lipschitz = False # lipschitz control
        
        # training loop
        length = epochs + (tparams.comp_relu if arch=="relu" else 0)

        for epoch in range(length):
            
            # train the model
            model.train()
            epoch_loss = 0
            epoch_start = datetime.now()

            # train over batches
            for X_batch, y_batch in train_loader:

                # forward pass + get loss
                y_pred = model(X_batch)
                if(arch == "relu" or arch == "both"):
                    optimizer.zero_grad()
                if(arch == "bspline" or arch == "both"):
                    aux_optimizer.zero_grad()
                    if lipschitz is True:
                        loss = lmbda * model.BV2()
                    else:
                        loss = lmbda * model.TV2()

                loss += loss_fn(y_pred, y_batch)
                epoch_loss += float(loss) * len(X_batch)

                # compute gradient and step on the optimizer
                loss.backward()
                if(arch == "relu" or arch == "both"):
                    optimizer.step()
                if(arch == "bspline" or arch == "both"):
                    aux_optimizer.step()                
            
            # step the LR scheduler
            if(tparams.lrs != "none" and tparams.lrs != "None"):
                if((arch == "bspline" or arch == "both") and scheduler.get_last_lr()[0] * tparams.lrs_gamma > 0.0001):
                    scheduler.step()
                if((arch=="bspline" or arch=="both") and aux_scheduler.get_last_lr()[0] * tparams.lrs_gamma > 0.00001):
                    aux_scheduler.step()

            # training loss: normalize to per-input MSE for saving
            avg_epoch_loss = epoch_loss / len(X_train)

            # validation loss (on the whole val dataset)
            model.eval()
            y_pred = model(X_test) # pass in all the validation data
            loss = float(loss_fn(y_pred, y_test)) #! could move this to after the next chunk to track times more accurately, but would be annoying to refactor for doing val loss after that
            
            if(tparams.comp_relu > 0 and arch == "relu"):
                in_pretraining = epoch < length - tparams.comp_relu - 1
                if(in_pretraining):
                    # if in pre-training, we want to append to both
                    train_history.append(avg_epoch_loss)
                    train_times.append(datetime.now() - epoch_start)
                    val_history.append(round(loss, 4))

                    cr_train_history.append(avg_epoch_loss)
                    cr_train_times.append(datetime.now() - epoch_start)
                    cr_val_history.append(round(loss, 4))
                if(not in_pretraining):
                    # if in comp-relu phase, just append to cr
                    cr_train_history.append(avg_epoch_loss)
                    cr_train_times.append(datetime.now() - epoch_start)
                    cr_val_history.append(round(loss, 4))
            else:
                train_history.append(avg_epoch_loss)
                train_times.append(datetime.now() - epoch_start)
                val_history.append(round(loss, 4))

            if(tparams.comp_relu > 0 and arch=="relu"): #
                if(epoch == length - tparams.comp_relu - 1): # if we're at the switch point, save the model
                    torch.save(model.state_dict(), f"./temp_models/{model_save_code}.pt")
        
        if(tparams.comp_relu == 0): # if not comp_relu, save at the end of the round of training
            torch.save(model.state_dict(), f"./temp_models/{model_save_code}.pt")
        
        elif(arch == "relu"): # if it is comp_relu and we're on relu 
            start_time = time.perf_counter()
            _ = model(X_test) # model output is irrelevant
            end_time = time.perf_counter()
            cr_fwd_lat = (end_time - start_time) / len(X_test) * 1000 * 1000 # per sample latency: seconds -> nanoseconds
            cr_fwd_lat = round(cr_fwd_lat, 4)

    # compute forward latency of end model

    #! NOT PER SAMPLE LATENCY THOUGH... X_test is 15% of dataset
    start_time = time.perf_counter()
    _ = model(X_test) # model output is irrelevant
    end_time = time.perf_counter()
    fwd_lat = (end_time - start_time) / len(X_test) * 1000 * 1000 # per sample latency: seconds -> nanoseconds
    fwd_lat = round(fwd_lat, 4)

    final_locs = None
    final_coeffs = None
    if(tparams.spline_epochs > 0):
        final_locs = model.get_deepspline_activations()[0]['locations']
        final_coeffs = model.get_deepspline_activations()[0]['coefficients']

    os.remove(f"./temp_models/{model_save_code}.pt")

    return model, train_history, val_history, train_times, fwd_lat, final_locs, final_coeffs, cr_val_history, cr_train_history, cr_train_times, cr_fwd_lat


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

    # comp relu
    cr_run_summaries = []
    cr_run_validations = [] # validations
    cr_run_history = [] # training 
    cr_run_epoch_times = []
    
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
        lrs_stepsize = cfg.lrs_stepsize,
        comp_relu = cfg.comp_relu,
    )
    
    for r in range(cfg.runs):
        start_time = datetime.now()
        model, train_history, val_history, train_times, fwd_lat, final_locs, final_coeffs, cr_val_history, cr_train_history, cr_train_times, cr_fwd_lat = training_run(mparams, tparams, X, y)

        run_validations.append(val_history)
        run_history.append(train_history)
        run_epoch_times.append([t.microseconds / 1000 / 1000 for t in train_times]) # us to seconds
        run_summaries.append([r+1, min(val_history), val_history[-1], round(sum(run_epoch_times[-1]),2), fwd_lat ])

        if(cfg.comp_relu > 0):
            cr_run_validations.append(cr_val_history)
            cr_run_history.append(cr_train_history)
            cr_run_epoch_times.append([t.microseconds / 1000 / 1000 for t in cr_train_times]) # us to seconds
            cr_run_summaries.append([r+1, min(cr_val_history), cr_val_history[-1], round(sum(cr_run_epoch_times[-1]),2), cr_fwd_lat ])

        print(f"Run {r} complete: {(datetime.now() - start_time).seconds}s")

    avg = sum(list(map(lambda r: r[2], run_summaries)))/len(run_summaries)
    if(cfg.comp_relu > 0):
        cr_avg = sum(list(map(lambda r: r[2], cr_run_summaries)))/len(cr_run_summaries)
    for r in range(len(run_summaries)):
        if(run_summaries[r][2] > avg*1.5 or (cfg.comp_relu > 0 and cr_run_summaries[r][2] > avg*1.5)):
            outliers += 1
            run_summaries[r].append(True) # replace w how close it is to avg
            if(cfg.comp_relu > 0):
                cr_run_summaries[r].append(True)
        else:
            run_summaries[r].append(False)
            if(cfg.comp_relu > 0):
                cr_run_summaries[r].append(False)
    
    rerun = 0
    if(outliers > 0):
        print(f"rerunning {outliers} outliers...")

    while(rerun < outliers):
        model, train_history, val_history, train_times, fwd_lat, final_locs, final_coeffs, cr_val_history, cr_train_history, cr_train_times, cr_fwd_lat = training_run(mparams, tparams, X, y)
        if(val_history[-1] <= avg*1.5):
            if(cfg.comp_relu > 0): # if comp_relu, we also need to check that
                if(cr_val_history[-1] <= cr_avg*1.5): 
                    run_validations.append(val_history)
                    run_history.append(train_history)
                    run_epoch_times.append([t.microseconds / 1000 / 1000 for t in train_times])
                    run_summaries.append([r+1, min(val_history), val_history[-1], round(sum(run_epoch_times[-1]),2), fwd_lat, False ])

                    cr_run_history.append(cr_train_history)
                    cr_run_epoch_times.append([t.microseconds / 1000 / 1000 for t in cr_train_times])
                    cr_run_summaries.append([r+1, min(cr_val_history), cr_val_history[-1], round(sum(cr_run_epoch_times[-1]),2), cr_fwd_lat ])
                    cr_run_summaries.append([r+1, min(cr_val_history), cr_val_history[-1], round(sum(cr_run_epoch_times[-1]),2), cr_fwd_lat, False])
            else:
                run_validations.append(val_history)
                run_history.append(train_history)
                run_epoch_times.append([t.microseconds / 1000 / 1000 for t in train_times])
                run_summaries.append([r+1, min(val_history), val_history[-1], round(sum(run_epoch_times[-1]),2), fwd_lat, False ])

            rerun += 1 
            print(f"Rerun outlier {rerun} complete")

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

    # time_deltas = [time_str_to_timedelta(row[3]) for row in run_summaries]
    time_deltas = [row[3] for row in run_summaries]
    fwd_latency_vals = [row[4] for row in run_summaries]
    average_time_delta = sum(time_deltas) / len(time_deltas)
    
    # average_time = timedelta_to_time_str(average_time_delta)
    average_fwd_latency = round(np.mean(fwd_latency_vals), 4)

    if(cfg.comp_relu > 0):
        cr_time_deltas = [row[3] for row in cr_run_summaries]
        cr_fwd_latency_vals = [row[4] for row in cr_run_summaries]
        cr_average_time_delta = sum(cr_time_deltas) / len(cr_time_deltas)
        
        # average_time = timedelta_to_time_str(average_time_delta)
        cr_average_fwd_latency = round(np.mean(cr_fwd_latency_vals), 4)

    data = {
        "mparams": mparams.to_dict(),
        "tparams": tparams.to_dict(),

        "flops (final model)": flops,
        "params (final model)": params,

        "avg_time": average_time_delta,
        "avg_fwdlat": average_fwd_latency,

        "times": run_epoch_times,
        "vals": run_validations,
        "trains": run_history,
    }
    if(cfg.comp_relu > 0):
        data.update({
            "avg_cr_time": cr_average_time_delta,
            "avg_cr_fwdlat": cr_average_fwd_latency,
            "cr_times": cr_run_epoch_times,
            "cr_vals": cr_run_validations,
            "cr_trains": cr_run_history,
        })

    if(tparams.lrs == "steplr"):
        out = f"{cfg.layers}_({cfg.lr_wb},{cfg.lr_bs},{cfg.lrs_gamma},{cfg.lrs_stepsize})_({cfg.relu_epochs},{cfg.spline_epochs},{cfg.both_epochs},{cfg.comp_relu})_{cfg.runs}"
    else:
        out = f"{cfg.layers}_({cfg.lr_wb},{cfg.lr_bs})_({cfg.relu_epochs},{cfg.spline_epochs},{cfg.both_epochs},{cfg.comp_relu})_{cfg.runs}"

    if(cfg.add_to_out != ""):
        out += "_" + cfg.add_to_out

    with open(f'saved/{out}.json', 'w') as f:
        json.dump(data, f)
    
    print(f"Saved to saved/{out}")

    return

if __name__ == "__main__":
    my_app()