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
    # cr_fwd_lat = -1

    model_save_code = ''.join(random.choices(string.ascii_letters + string.digits, k=8))

    for arch in ["relu", tparams.bspline_order[0], tparams.bspline_order[1], "lspline"]:
        
        epochs = 0

        #^ initialize the correct architecture
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
                if(tparams.bspline_epochs == 0):
                    continue
                if(tparams.bspline_order[0]=="bspline" or tparams.both_epochs==0):
                    # bspline before both (or no both)

                    if(tparams.relu_epochs > 0): # if we pretrained on ReLU, load those weights
                        model = LinearBSpline(mparams.layers, mparams.cpoints, mparams.range_)
                        model.load_state_dict(torch.load(f"./temp_models/{model_save_code}.pt", weights_only=True), strict=False)
                        # both went first
                    else:
                        model = LinearBSpline(mparams.layers, mparams.cpoints, mparams.range_)
                epochs = tparams.bspline_epochs
            if(arch == "both"):
                if(tparams.both_epochs == 0):
                    continue
                elif(tparams.bspline_order[0]=="both" or tparams.bspline_epochs==0):
                    # both before bspline or no bspline

                    if(tparams.relu_epochs > 0): # if we pretrained on ReLU, load those weights
                        model = LinearBSpline(mparams.layers, mparams.cpoints, mparams.range_)
                        model.load_state_dict(torch.load(f"./temp_models/{model_save_code}.pt", weights_only=True), strict=False)
                    else:
                        model = LinearBSpline(mparams.layers, mparams.cpoints, mparams.range_)
                epochs = tparams.both_epochs

            optimizer = optim.Adam(model.parameters_no_deepspline(), lr=tparams.lr_wb)
            aux_optimizer = optim.Adam(model.parameters_deepspline(), lr=tparams.lr_bs)

            if(tparams.lrs == "steplr"):
                # resetting scheduler
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=tparams.lrs_stepsize, gamma=tparams.lrs_gamma)
                aux_scheduler = torch.optim.lr_scheduler.StepLR(aux_optimizer, step_size=tparams.lrs_stepsize, gamma=tparams.lrs_gamma)
            
            lmbda = 1e-4 # regularization weight
            lipschitz = False # lipschitz control
        elif(arch == "lspline"):
            if(tparams.lspline_epochs == 0):
                continue
            if(tparams.bspline_epochs > 0 or tparams.both_epochs > 0):
                epochs = tparams.lspline_epochs
                bspline_layers=model.get_layers()
                model = linspline.LSplineFromBSpline(bspline_layers)
                optimizer = optim.Adam(model.parameters(), lr=tparams.lr_ls)
            else:
                continue
        
        train_wb = arch in ["relu", "both", "lspline"] # train weights and biases
        train_af = arch in ["bspline", "both"] # train activation func

        length = epochs + (tparams.comp_relu if arch=="relu" else 0)
        if length == 0:
            continue # skip rounds on 0 epochs

        # ^ training loop
        model.train()
        for epoch in range(length):

            epoch_loss = 0
            epoch_start = datetime.now()
            # train over batches
            for X_batch, y_batch in train_loader:

                # zero grad, forward pass, and calc loss
                y_pred = model(X_batch)

                if(train_wb):
                    optimizer.zero_grad()

                loss = loss_fn(y_pred, y_batch)
                epoch_loss += float(loss) * len(X_batch) # dont record bspline loss (for comparison)

                if(train_af):
                    aux_optimizer.zero_grad()
                    loss += lmbda * (model.BV2() if lipschitz else model.TV2())

                # compute gradient and step on the optimizer
                loss.backward()

                if(train_wb):
                    optimizer.step()
                if(train_af):
                    aux_optimizer.step()    

            epoch_end = datetime.now()      
            
            # step the LR scheduler
            if(tparams.lrs != "none" and tparams.lrs != "None"):
                if(train_wb and scheduler.get_last_lr()[0] * tparams.lrs_gamma > 0.0001):
                    scheduler.step()
                if(train_af and aux_scheduler.get_last_lr()[0] * tparams.lrs_gamma > 0.00001):
                    aux_scheduler.step()
            
            # training loss: normalize to per-input MSE for saving
            avg_epoch_loss = epoch_loss / len(X_train)

            # validation loss (on the whole val dataset)
            model.eval()
            y_pred = model(X_test) # pass in all the validation data
            loss = float(loss_fn(y_pred, y_test)) # validation loss on the whole dataset
            
            if(tparams.comp_relu > 0 and arch == "relu"):
                in_pretraining = epoch < length - tparams.comp_relu - 1
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

            if(tparams.comp_relu > 0 and arch=="relu" and epoch == length - tparams.comp_relu - 1): #
                # if we're (using comp_relu) and at the switch point, save the model
                torch.save(model.state_dict(), f"./temp_models/{model_save_code}.pt")
        
        # after training
        if(arch == "relu" and tparams.relu_epochs > 0):
            if(tparams.comp_relu == 0): # if not comp_relu, save at the end of the round of training
                torch.save(model.state_dict(), f"./temp_models/{model_save_code}.pt")
    
    if(tparams.relu_epochs > 0): # this is only used if we pretrained on ReLU
        os.remove(f"./temp_models/{model_save_code}.pt")

    train_times2 = [t.seconds+(t.microseconds / 1000 / 1000) for t in train_times]
    cr_train_times2 = [t.seconds+(t.microseconds / 1000 / 1000) for t in cr_train_times]
    
    return model, train_history, val_history, train_times2, cr_val_history, cr_train_history, cr_train_times2


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
        bspline_epochs = cfg.bspline_epochs,
        both_epochs = cfg.both_epochs,
        lspline_epochs = cfg.lspline_epochs,
        batch_size = cfg.batch_size,
        lr_wb = cfg.lr_wb,
        lr_bs = cfg.lr_bs,
        lr_ls = cfg.lr_ls,
        lrs = cfg.lrs,
        lrs_gamma = cfg.lrs_gamma,
        lrs_stepsize = cfg.lrs_stepsize,
        comp_relu = cfg.comp_relu,
        bspline_order = cfg.bspline_order,
    )

    print("Init complete.")
    
    for r in range(cfg.runs):
        start_time = datetime.now()
        model, train_history, val_history, train_times, cr_val_history, cr_train_history, cr_train_times = training_run(mparams, tparams, X, y)

        run_validations.append(val_history)
        run_history.append(train_history)
        run_epoch_times.append(train_times) # microsec to seconds
        run_summaries.append([r+1, min(val_history), val_history[-1], round(sum(run_epoch_times[-1]),2) ])

        if(cfg.comp_relu > 0):
            cr_run_validations.append(cr_val_history)
            cr_run_history.append(cr_train_history)
            cr_run_epoch_times.append(cr_train_times) # microsec to seconds
            cr_run_summaries.append([r+1, min(cr_val_history), cr_val_history[-1], round(sum(cr_run_epoch_times[-1]),2) ])

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
        model, train_history, val_history, train_times, cr_val_history, cr_train_history, cr_train_times = training_run(mparams, tparams, X, y)
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
    }
    
    if(cfg.comp_relu > 0):
        data.update({
            "avg_cr_time": cr_average_time_delta,
            "cr_times": cr_run_epoch_times,
            "cr_vals": cr_run_validations,
            "cr_trains": cr_run_history,
        })

    if(tparams.lrs == "steplr"):
        out = f"{cfg.layers}_({cfg.lr_wb},{cfg.lr_bs},{cfg.lr_ls},{cfg.lrs_gamma},{cfg.lrs_stepsize})_({cfg.relu_epochs},{cfg.bspline_epochs},{cfg.both_epochs},{cfg.comp_relu},{cfg.lspline_epochs})_{cfg.runs}"
    else:
        out = f"{cfg.layers}_({cfg.lr_wb},{cfg.lr_bs},{cfg.lr_ls})_({cfg.relu_epochs},{cfg.bspline_epochs},{cfg.both_epochs},{cfg.comp_relu},{cfg.lspline_epochs})_{cfg.runs}"

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