# regression 8 adds support for bike sharing dataset

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from datetime import datetime

from omegaconf import DictConfig, OmegaConf, ListConfig
import hydra

from models import LinearReLU, LinearBSpline

import json
from datetime import datetime
from torch.utils.data import TensorDataset, DataLoader

import linspline

import string, random, os, re
import pandas as pd

import sys, os
sys.path.insert(0, os.path.abspath('../DeepSplines'))
import deepsplines

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


def training_run(mparams, tparams, X, y, loss_fn):
    
    # train-test split of the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.85, shuffle=True)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

    # using a dataloader to randomize batching
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=tparams.batch_size, shuffle=True)

    train_history = []
    val_history = []
    train_times = []

    # comp relu
    cr_val_history = []
    cr_train_history = []
    cr_train_times = []

    model_save_code = ''.join(random.choices(string.ascii_letters + string.digits, k=8))

    arch = ""

    lmbda = 1e-4 # regularization weight
    lipschitz = False # lipschitz control
    model = None

    isfirstepoch = True
    alternate_epoch_training = True #! temp

    for train_arch in tparams.epoch_specs:
        '''
            xxxR = xxx ReLU epochs
            xxxB = xxx BSpline epochs (classical weights frozen)
            xxxFB = xxx Frozen BSpline epochs (classical weights active, BSpline weights frozen)
            xxxWB = xxx BSpline + classical weights epochs
            xxxL = xxx LinSpline epochs
        '''

        arch = re.findall(r"[A-Za-z]+", train_arch)[0]
        epochs = int(re.findall(r'\d+', train_arch)[0])
        from_relu = False

        if(arch == "R"): # ReLU epochs, can ONLY come first
            model = LinearReLU(mparams.layers)
            optimizer = optim.Adam(model.parameters(), lr=tparams.lr_wb)
            if(tparams.lrs == "steplr"):
                    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=tparams.lrs_stepsize, gamma=tparams.lrs_gamma)
            epochs += tparams.comp_relu

        elif(arch != "L"): # one of the BSpline models: WB, FB, or B
            if(type(model) == LinearReLU): # ReLU to a BSpline
                model = LinearBSpline(mparams.layers, mparams.cpoints, mparams.range_)
                model.load_state_dict(torch.load(f"./temp_models/{model_save_code}.pt", weights_only=True), strict=False)
                from_relu = True
            elif(model is None): # no ReLU pretraining
                model = LinearBSpline(mparams.layers, mparams.cpoints, mparams.range_)

            if(arch == "FB"):
                for wb_param in model.parameters_no_deepspline():
                    wb_param.requires_grad = True
                for af_param in model.parameters_deepspline():
                    af_param.requires_grad = False
                optimizer = optim.Adam(model.parameters_no_deepspline(), lr=tparams.lr_fb)
                aux_optimizer = None

            elif(arch == "WB"):
                for wb_param in model.parameters_no_deepspline():
                    wb_param.requires_grad = True
                for af_param in model.parameters_deepspline():
                    af_param.requires_grad = True

                if(isfirstepoch or from_relu):
                    optimizer = optim.Adam(model.parameters_no_deepspline(), lr=tparams.lr_wb)
                else:
                    optimizer = optim.Adam(model.parameters_no_deepspline(), lr=tparams.lr_transfer_WBS)
                aux_optimizer = optim.Adam(model.parameters_deepspline(), lr=tparams.lr_bs)


                if(tparams.lrs == "steplr-wbstransfer" and tparams.lrs == "steplr-wbstransfer"):
                    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=tparams.lrs_stepsize, gamma=tparams.lrs_gamma)
                
            elif(arch == "B"):
                for wb_param in model.parameters_no_deepspline():
                    wb_param.requires_grad = False
                for af_param in model.parameters_deepspline():
                    af_param.requires_grad = True
                optimizer = None
                aux_optimizer = optim.Adam(model.parameters_deepspline(), lr=tparams.lr_bs)

        else: # LSpline epochs, can ONLY come last
            if(type(model) == LinearReLU):
                model = LinearBSpline(mparams.layers, mparams.cpoints, mparams.range_)
                model.load_state_dict(torch.load(f"./temp_models/{model_save_code}.pt", weights_only=True), strict=False)
            
            model = linspline.LSplineFromBSpline(model.get_layers())
            optimizer = optim.Adam(model.parameters(), lr=tparams.lr_ls)
            aux_optimizer = None
        
        isfirstepoch = False
                
        train_wb = arch in ["R", "WB", "FB", "L"] # train weights and biases
        train_af = arch in ["WB", "B"] # train activation func
        
        # training loop
        for epoch in range(epochs):

            model.train()
            epoch_loss = 0
            epoch_start = datetime.now()
            
            # train over batches
            for X_batch, y_batch in train_loader:

                # zero grad, forward pass, and calc loss
                y_pred = model(X_batch)

                if(train_wb and ((not alternate_epoch_training) or epoch%2==0)):
                    optimizer.zero_grad()

                loss = loss_fn(y_pred, y_batch)
                epoch_loss += float(loss) * len(X_batch) # dont record bspline loss (for comparison)

                if(train_af and ((not alternate_epoch_training) or epoch%2==1)):
                    aux_optimizer.zero_grad()
                    loss += lmbda * (model.BV2() if lipschitz else model.TV2())

                # compute gradient and step on the optimizer
                loss.backward()

                if(train_wb and ((not alternate_epoch_training) or epoch%2==0)):
                    optimizer.step()
                if(train_af and ((not alternate_epoch_training) or epoch%2==1)):
                    aux_optimizer.step()    

            epoch_end = datetime.now()      
            
            # training loss: normalize to per-input MSE for saving
            avg_epoch_loss = epoch_loss / len(X_train)

            # validation loss (on the whole val dataset)
            model.eval()
            y_pred = model(X_test) # pass in all the validation data
            loss = float(loss_fn(y_pred, y_test)) # validation loss on the whole dataset

            if(tparams.lrs == "steplr" or (arch=="WB" and tparams.lrs == "steplr-wbstransfer")):
                if(train_wb):
                    scheduler.step()
                # if(train_af and aux_scheduler.get_last_lr()[0] * tparams.lrs_gamma > 0.00001):
                    # aux_scheduler.step()
            
            if(tparams.comp_relu > 0 and arch == "R"):
                in_pretraining = epoch < epochs - tparams.comp_relu - 1
                if(in_pretraining):
                    # if in pre-training, we want to append to both
                    train_history.append(avg_epoch_loss)
                    train_times.append(epoch_end - epoch_start)
                    val_history.append(round(loss, 4))

                    cr_train_history.append(avg_epoch_loss)
                    cr_train_times.append(epoch_end - epoch_start)
                    cr_val_history.append(round(loss, 4))
                if(not in_pretraining):
                    # if in comp-relu phase, just append to cr
                    cr_train_history.append(avg_epoch_loss)
                    cr_train_times.append(epoch_end - epoch_start)
                    cr_val_history.append(round(loss, 4))
            else:
                train_history.append(avg_epoch_loss)
                train_times.append(epoch_end - epoch_start)
                val_history.append(round(loss, 4))

            if(tparams.comp_relu > 0 and arch=="R" and epoch == epochs - tparams.comp_relu - 1): #
                # if we're (using comp_relu) and at the switch point, save the model
                torch.save(model.state_dict(), f"./temp_models/{model_save_code}.pt")

        if(tparams.comp_relu == 0 and arch == "R"): # otherwise save at the end of ReLU training
            torch.save(model.state_dict(), f"./temp_models/{model_save_code}.pt")
    
    if(tparams.epoch_specs[0][-1] == "R" and tparams.comp_relu > 0): # this is only used if we pretrained on ReLU
        os.remove(f"./temp_models/{model_save_code}.pt")

    train_times2 = [t.seconds+(t.microseconds / 1000 / 1000) for t in train_times]
    cr_train_times2 = [t.seconds+(t.microseconds / 1000 / 1000) for t in cr_train_times]

    return model, train_history, val_history, train_times2, cr_val_history, cr_train_history, cr_train_times2


import utils
@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg: DictConfig) -> None:

    if(cfg.runs == 0):
        return
    
    print("Parameters: ")
    print(OmegaConf.to_yaml(cfg))

    # By default, PyTorch attempts to use all available CPU cores for intra-op parallelism. Set threads = cpu cores
    torch.set_num_threads(cfg.threads)

    # Load the data

    if(cfg.dataset == "cal_housing"):
        housing = fetch_california_housing()
        loss_fn = nn.MSELoss()  # mean square error
        X, y = housing.data, housing.target
    # elif(cfg.dataset == "bike_sharing"):
    #     bikes = pd.read_csv('bikesharing/day.csv') # not spatial
    #     filtered = bikes.copy()
    #     remove = ["dteday", "yr", "instant", "casual", "registered", "season", "workingday"]
    #     for column in remove:
    #         filtered = filtered.drop(column, axis=1)
    #     data = filtered.drop("cnt", axis=1)
    #     target = filtered["cnt"]
    #     X, y = data.values, target.values
    #     loss_fn = nn.RMSELoss()  # root mean squared error
    else:
        print("invalid dataset")

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
        epoch_specs = cfg.epoch_specs,
        batch_size = cfg.batch_size,
        lr_wb = cfg.lr_wb,
        lr_bs = cfg.lr_bs,
        lr_ls = cfg.lr_ls,
        lr_fb = cfg.lr_fb,
        lr_transfer_WBS = cfg.lr_transfer_WBS,
        lrs = cfg.lrs,
        lrs_gamma = cfg.lrs_gamma,
        lrs_stepsize = cfg.lrs_stepsize,
        comp_relu = cfg.comp_relu,
    )

    print("Init complete.")
    
    zeroed = []
    for r in range(cfg.runs):

        start_time = datetime.now()
        model, train_history, val_history, train_times, cr_val_history, cr_train_history, cr_train_times = training_run(mparams, tparams, X, y, loss_fn)
        
        run_validations.append(val_history)
        run_history.append(train_history)
        run_epoch_times.append(train_times) # microsec to seconds
        run_summaries.append([r+1, min(val_history), val_history[-1], round(sum(run_epoch_times[-1]),2) ])

        if(cfg.comp_relu > 0):
            cr_run_validations.append(cr_val_history)
            cr_run_history.append(cr_train_history)
            cr_run_epoch_times.append(cr_train_times) # microsec to seconds
            cr_run_summaries.append([r+1, min(cr_val_history), cr_val_history[-1], round(sum(cr_run_epoch_times[-1]),2) ])

        zeroed.append(utils.get_model_zeroed_activations(model))

        print(f"Run {r} complete: {(datetime.now() - start_time).seconds}s")

    avg_fin_loss = sum(list(map(lambda r: r[2], run_summaries)))/len(run_summaries)
    if(cfg.comp_relu > 0):
        cr_avg = sum(list(map(lambda r: r[2], cr_run_summaries)))/len(cr_run_summaries)
    for r in range(len(run_summaries)):
        if(run_summaries[r][2] > avg_fin_loss*1.5 or (cfg.comp_relu > 0 and cr_run_summaries[r][2] > avg_fin_loss*1.5)):
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
        model, train_history, val_history, train_times, cr_val_history, cr_train_history, cr_train_times = training_run(mparams, tparams, X, y, loss_fn)
        if(val_history[-1] <= avg_fin_loss*1.5):
            if(cfg.comp_relu > 0 and cr_val_history[-1] <= cr_avg*1.5): # if comp_relu, we also need to check that
                run_validations.append(val_history)
                run_history.append(train_history)
                run_epoch_times.append(train_times)
                run_summaries.append([r+1, min(val_history), val_history[-1], round(sum(run_epoch_times[-1]),2), False ])

                cr_run_validations.append(cr_val_history)
                cr_run_history.append(cr_train_history)
                cr_run_epoch_times.append(cr_train_times)
                cr_run_summaries.append([r+1, min(cr_val_history), cr_val_history[-1], round(sum(cr_run_epoch_times[-1]),2), False])
            else:
                run_validations.append(val_history)
                run_history.append(train_history)
                run_epoch_times.append(train_times)
                run_summaries.append([r+1, min(val_history), val_history[-1], round(sum(run_epoch_times[-1]),2), False ])

            rerun += 1 
            print(f"Rerun outlier {rerun} complete")

    time_deltas = [row[3] for row in run_summaries]
    average_time_delta = sum(time_deltas) / len(time_deltas)

    if(cfg.comp_relu > 0):
        cr_time_deltas = [row[3] for row in cr_run_summaries]
        cr_average_time_delta = sum(cr_time_deltas) / len(cr_time_deltas)

    data = {
        "mparams": mparams.to_dict(),
        "tparams": tparams.to_dict(),

        "avg_time": average_time_delta,
        "times": run_epoch_times,
        "vals": run_validations,
        "trains": run_history,
        "zeroed": zeroed, # doesn't factor in outliers
    }
    
    if(cfg.comp_relu > 0):
        data.update({
            "avg_cr_time": cr_average_time_delta,
            "cr_times": cr_run_epoch_times,
            "cr_vals": cr_run_validations,
            "cr_trains": cr_run_history,
        })

    if(tparams.lrs == "steplr"):
        out = f"{cfg.layers}_({cfg.lr_wb},{cfg.lr_bs},{cfg.lr_ls},{cfg.lrs_gamma},{cfg.lrs_stepsize})_({cfg.epoch_specs})_{cfg.runs}"
    else:
        out = f"{cfg.layers}_({cfg.lr_wb},{cfg.lr_bs},{cfg.lr_ls})_({cfg.epoch_specs})_{cfg.runs}"

    if(cfg.bspline_order == ["both","bspline"]):
        out += "_bothfirst"

    if(cfg.cpoints != 3):
        out += f"_{cfg.cpoints}"

    if(cfg.add_to_out != ""):
        out += "_" + cfg.add_to_out

    if not os.path.exists(f'saved/{cfg.output_dir}'):
        os.makedirs(f'saved/{cfg.output_dir}')
        
    with open(f'saved/{cfg.output_dir}/{out}.json', 'w') as f:
        json.dump(data, f)

    print(f"Saved to saved/{cfg.output_dir}/{out}")

    return

if __name__ == "__main__":
    my_app()