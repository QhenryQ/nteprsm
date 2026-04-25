import os
import shutil
import argparse
import pickle
from datetime import datetime
import numpy as np
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
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%y")
    df["rating_event"] = df["rater"] + '-' + df["date"].dt.strftime("%m-%d-%y")
    df["rater_code"] = pd.Categorical(df["rater"]).codes + 1
    df["rating_event_code"] = pd.Categorical(df["rating_event"]).codes + 1
    
    # create data handler
    datahandler = utils.DataHandler(filepath=config["data_path"])
    datahandler.load_data(df=df)
    datahandler.preprocess_data()
    
    # train/test split: hold out ~20% of rating events, approximately evenly spaced over the event index
    unique_events = (
        datahandler.model_data[["rating_event", "date"]]
        .drop_duplicates()
        .sort_values(["date", "rating_event"])
        .reset_index(drop=True)
    )
    num_events = len(unique_events)
    num_held_out = max(1, round(num_events * 0.2))
    held_out_idx = np.round(np.linspace(1, num_events, num_held_out + 2))[1:-1].astype(int)
    held_out_events = unique_events.iloc[held_out_idx]["rating_event"].tolist()
    train_df = datahandler.model_data[~datahandler.model_data["rating_event"].isin(held_out_events)].copy()
    test_df = datahandler.model_data[datahandler.model_data["rating_event"].isin(held_out_events)].copy()
    # re-encode rating_event_code to be continuous after removing held-out events
    train_df["rating_event_code"] = pd.Categorical(train_df["rating_event"]).codes + 1
    datahandler.model_data = train_df
    logger.info(
        f"Train/test split: {len(train_df)} train rows, {len(test_df)} test rows "
        f"({len(held_out_events)} held-out rating events: {held_out_events})."
    )
    
    datahandler.generate_stan_data(**config["stan_additional_data"])
    
    # re-encode test rating events to be continuous and inject into stan_data
    test_df["rating_event_code_test"] = pd.Categorical(test_df["rating_event"]).codes + 1
    time_test = (
        test_df[["rating_event_code_test", "adj_time_of_year"]]
        .drop_duplicates()
        .sort_values("rating_event_code_test")
        .set_index("rating_event_code_test")
        .values.reshape(-1)
    )
    datahandler.stan_data.update({
        "N_test": len(test_df),
        "y_test": test_df[datahandler.target].values,
        "num_ratings_test": int(test_df["rating_event_code_test"].max()),
        "time_test": time_test,
        "entry_code_test": test_df["entry_code"].values,
        "plot_code_test": test_df["plot_code"].values,
        "rater_code_test": test_df["rater_code"].values,
        "rating_event_code_test": test_df["rating_event_code_test"].values,
    })
    print ('Train dates: ', train_df['date'].unique().tolist())
    print ('Test dates: ', test_df['date'].unique().tolist())
    
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
    # Save the model
    with open(working_dir + "/" + model_output_file, 'wb') as file:
        pickle.dump(fit, file)
    # Save witheld test data for future assessment 
    with open(working_dir + "/test_data.pkl", 'wb') as file:
        pickle.dump(test_df, file)


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
    #to execute
    #python nteprsm/model_umn_with_split.py config/annual_seasonality_model_with_split.yml kb2017/nj2/quality.csv model_runs/seasonality_nj2_cross_val fit_seasonality_nj2_cross_val.pkl 
