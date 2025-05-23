import os
import shutil
import argparse
import pickle
from datetime import datetime

import pandas as pd
from cmdstanpy import CmdStanModel

from gptools.stan import get_include
from nteprsm import utils
from settings import CONFIG_DIR


def main(config_file: str, data_path: str, working_dir: str, model_output_file: str):
    
    os.makedirs(working_dir, exist_ok=True)
    logger = utils.setup_logging(working_dir)
    
    config = utils.load_config(config_file)
    config["data_path"] = data_path
    
    # process data
    df = pd.read_csv(config["data_path"])  # Replace 'file.csv' with your file path
    # convert all code to 1-indexed as stan is 1-indexed
    df["rating_event"] = df["rater"] + '-' + pd.to_datetime(df["date"], format="%m/%d/%y").dt.strftime("%m-%d-%y")
    df["rater_code"] = pd.Categorical(df["rater"]).codes + 1
    df["rating_event_code"] = pd.Categorical(df["rating_event"]).codes + 1
    
    # cretae data handler
    datahandler = utils.DataHandler(filepath=config["data_path"])
    datahandler.load_data(raw_data=df)
    datahandler.preprocess_data()
    datahandler.generate_stan_data(**config["stan_additional_data"])
    
    shutil.copy2(config["stan_file"], working_dir)
    config["sampling"]["output_dir"] = working_dir
    
    # compile model executable
    nteprsm = CmdStanModel(
        stan_file=working_dir + "/" + config["stan_file"].split("/")[-1],
        stanc_options={"include-paths": get_include()}
    )
    
    starttime = datetime.now()
    logger.info("Model fit starts now!!!")
    # samples will be saved in the csv files in the output directory specified in the config
    fit = nteprsm.sample(data=datahandler.stan_data, **config["sampling"])  
    endtime = datetime.now()
    logger.info(f"Model fit ended, it took {(endtime - starttime).total_seconds()/60} minutes.")
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
