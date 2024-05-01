import argparse
from cmdstanpy import CmdStanModel
from gptools.stan import get_include

from nteprsm import utils
from settings import CONFIG_DIR, LOG_DIR

logger = utils.setup_logging(LOG_DIR)

def main(config_file: str):
    # model config
    config = utils.load_config(config_file)
    # process data
    datahandler = utils.DataHandler(
        filepath=config["data_path"], logger=logger
    )
    datahandler.load_data()
    datahandler.preprocess_data()
    datahandler.generate_stan_data(**config["stan_additional_data"])

    # model fitting
    nteprsm = CmdStanModel(
        stan_file=config["stan_file"],
        stanc_options={"include-paths": get_include()},
    )
    # samples will be saved in the csv files in the output directory specified in the config
    fit = nteprsm.sample(data=datahandler.stan_data, **config["sampling"])

def parse_args():
    parser = argparse.ArgumentParser(description="Run the Stan model fitting process.")
    parser.add_argument("config_file", type=str, help=f"Configuration file name located in {CONFIG_DIR}")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args.config_file)
