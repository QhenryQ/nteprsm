import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel
import numpy as np
from gptools.stan import get_include
from nteprsm import utils 
from settings import CONFIG_DIR, LOG_DIR
import os
import shutil
from datetime import datetime
import argparse

import pickle


def main(config_file: str, data_path: str, working_dir: str, model_output_file: str):
    
    os.makedirs(working_dir, exist_ok=True)
    logger = utils.setup_logging(working_dir)
    
    config = utils.load_config(config_file)
    config["data_path"] = data_path
    
    # process data
    df = pd.read_csv(config["data_path"])  # Replace 'file.csv' with your file path
    df.columns = [col.lower() for col in df.columns]
    df["rating_event"] = df["rater"] +  '-' + pd.to_datetime(df['date'], format="%m/%d/%y").dt.strftime('%m-%d-%y')
    df = df.assign(
    entry_name_code=pd.Categorical(df["entry_name"]).codes,
    plt_id_code=pd.Categorical(df["plt_id"]).codes,
    rater_code=pd.Categorical(df["rater"]).codes,
    rating_event_code=pd.Categorical(df["rating_event"]).codes,
    )
    df["entry_cumcount"] = df.groupby("entry_name").cumcount() + 1
    # cretae data handler
    datahandler = utils.DataHandler(filepath=config["data_path"])
    datahandler.load_data()
    datahandler.raw_data = df
    print (datahandler.raw_data.columns)
    datahandler.preprocess_data()
    datahandler.generate_stan_data(**config["stan_additional_data"])
    
    id_code_to_plt_id = datahandler.model_data.groupby('plt_id_code')['plt_id'].mean().astype(int).to_dict()
    plt_id_to_row =  datahandler.model_data.groupby('plt_id')['row'].mean().astype(int).to_dict()
    plt_id_to_col = datahandler.model_data.groupby('plt_id')['col'].mean().astype(int).to_dict()

    num_plots = datahandler.stan_data['num_plots']
    dist_matrix = np.zeros(shape = (num_plots, num_plots))
    for i in range(num_plots):
        for j in range(num_plots):
            plt_id_i, plt_id_j = id_code_to_plt_id[i], id_code_to_plt_id[j]
            row_i, row_j = plt_id_to_row[plt_id_i], plt_id_to_row[plt_id_j]
            col_i, col_j = plt_id_to_col[plt_id_i], plt_id_to_col[plt_id_j]
            dist = np.sqrt((row_i - row_j)**2 + (col_i - col_j)**2)
            dist_matrix[i][j] = dist
    
    # rename variables
    datahandler.stan_data["I"] = len(datahandler.model_data['rating_event_code'].unique())
    datahandler.stan_data["J"] = datahandler.stan_data['num_entries']
    datahandler.stan_data["P"] = datahandler.stan_data['num_plots']
    datahandler.stan_data["M"] = datahandler.stan_data['num_categories'] - 1
    datahandler.stan_data["ii"] = datahandler.model_data['rating_event_code'] + 1
    datahandler.stan_data["jj"] = datahandler.model_data['entry_name_code'] + 1
    datahandler.stan_data["pp"] = datahandler.model_data['plt_id_code'] + 1
    #datahandler.stan_data["y"] =  datahandler.model_data['quality'] - 1
    #datahandler.stan_data["y"] =  datahandler.model_data['value'] - 1

    # create dist matrix
    id_code_to_plt_id = datahandler.model_data.groupby('plt_id_code')['plt_id'].mean().astype(int).to_dict()
    plt_id_to_row =  datahandler.model_data.groupby('plt_id')['row'].mean().astype(int).to_dict()
    plt_id_to_col = datahandler.model_data.groupby('plt_id')['col'].mean().astype(int).to_dict()

    num_plots = datahandler.stan_data['num_plots']
    dist_matrix = np.zeros(shape = (num_plots, num_plots))
    for i in range(num_plots):
        for j in range(num_plots):
            plt_id_i, plt_id_j = id_code_to_plt_id[i], id_code_to_plt_id[j]
            row_i, row_j = plt_id_to_row[plt_id_i], plt_id_to_row[plt_id_j]
            col_i, col_j = plt_id_to_col[plt_id_i], plt_id_to_col[plt_id_j]
            dist = np.sqrt((row_i - row_j)**2 + (col_i - col_j)**2)
            dist_matrix[i][j] = dist
    datahandler.stan_data["DIST"] = dist_matrix
    
    shutil.copy2(config["stan_file"], working_dir)
    
    config["sampling"]["output_dir"] = working_dir
    
    # compile model executable
    nteprsm = CmdStanModel(
        stan_file=working_dir + "/" + config["stan_file"].split("/")[-1],
        stanc_options={"include-paths": get_include()}
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
