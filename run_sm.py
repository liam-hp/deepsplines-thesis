from omegaconf import DictConfig
import hydra
import utils

@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg: DictConfig) -> None:

    utils.train_models_count_zeroed_AFs(
        save_output=cfg.add_to_out,
        runs = 5,
        layers=cfg.layers, 
        epochs={"relu": 200, "both": 0, "bspline": 10}
    )

if __name__ == "__main__":
    my_app()