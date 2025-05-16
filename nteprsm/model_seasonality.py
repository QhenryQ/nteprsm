import os
import shutil
from datetime import datetime

import argparse
from cmdstanpy import CmdStanModel
from gptools.stan import get_include

from nteprsm import utils
from settings import CONFIG_DIR, LOG_DIR

import pandas as pd
import pickle


def main(config_file: str, data_path: str, working_dir: str, model_output_file: str):
    
    os.makedirs(working_dir, exist_ok=True)
    logger = utils.setup_logging(working_dir)
    
    # model config
    config = utils.load_config(config_file)
    config["sampling"]["output_dir"] = working_dir
    config["data_path"] = data_path
    # process data
    datahandler = utils.DataHandler(
        filepath=config["data_path"], logger=logger
    )
    datahandler.load_data()
    if 'plt_id' not in datahandler.raw_data.columns:
        datahandler.raw_data['plt_id'] = datahandler.raw_data['plt_code']
    if 'entry_name' not in datahandler.raw_data.columns:
        datahandler.raw_data['entry_name'] = datahandler.raw_data['entry_code']
    if 'rating_event' not in datahandler.raw_data.columns:
        datahandler.raw_data['rating_event'] = datahandler.raw_data['rater'] +  '-' + pd.to_datetime(datahandler.raw_data['date'], format="%m/%d/%y").dt.strftime('%m-%d-%y')
    datahandler.preprocess_data()
    datahandler.generate_stan_data(**config["stan_additional_data"])
    
    shutil.copy2(config["stan_file"], working_dir)
    
    # model fitting
    nteprsm = CmdStanModel(
        stan_file=working_dir + "/" + config["stan_file"].split("/")[-1],
        stanc_options={"include-paths": get_include()},
    )
    StartTime = datetime.now()
    logger.info("Model fit starts now!!!")
    # samples will be saved in the csv files in the output directory specified in the config
    fit = nteprsm.sample(data=datahandler.stan_data, **config["sampling"])
    EndTime = datetime.now()
    logger.info(f"Model fit ended, it took {EndTime - StartTime}")
    #Save the model
    with open(working_dir + "/" + model_output_file, 'wb') as file:
        pickle.dump(fit, file)
    
def parse_args():
    parser = argparse.ArgumentParser(description="Run the Stan model fitting process.")
    parser.add_argument("config_file", type=str, help=f"Configuration file name located in {CONFIG_DIR}")
    parser.add_argument("data_path", type=str, help=f"File to process")
    parser.add_argument("working_dir", type=str, help=f"Working directory for this run")
    parser.add_argument("model_output_file", type=str, help=f"Output pickle file")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args.config_file, args.data_path, args.working_dir, args.model_output_file)
