import os
import shutil
import argparse
import pickle
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel

from gptools.stan import get_include
from nteprsm import utils
from settings import CONFIG_DIR


def _load_held_out_dates(holdout_test_data_path: str | None) -> list[str] | None:
    if holdout_test_data_path is None:
        return None

    holdout_path = Path(holdout_test_data_path)
    if holdout_path.suffix == ".pkl":
        with holdout_path.open("rb") as file:
            held_out_df = pickle.load(file)
    else:
        held_out_df = pd.read_csv(holdout_path)

    if "date" not in held_out_df.columns:
        raise ValueError("Held-out data must include a 'date' column.")

    held_out_dates = (
        pd.to_datetime(held_out_df["date"])
        .dt.strftime("%Y-%m-%d")
        .dropna()
        .unique()
        .tolist()
    )
    if not held_out_dates:
        raise ValueError("Held-out data did not contain any valid dates.")
    return sorted(held_out_dates)


def main(
    config_file: str,
    data_path: str,
    working_dir: str,
    model_output_file: str,
    holdout_test_data_path: str | None = None,
):
    
    os.makedirs(working_dir, exist_ok=True)
    logger = utils.setup_logging(working_dir)
    
    config = utils.load_config(config_file)
    config["data_path"] = data_path
    
    # process data
    df = pd.read_csv(config["data_path"])  # Replace 'file.csv' with your file path
    df.columns = df.columns.str.lower()
    df = df.rename(columns={"quality": "value", "ploc_code": "plot_code"})

    # convert all code to 1-indexed as stan is 1-indexed
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%y")
    # Treat each calendar date as one rating event so held-out validation uses unseen dates.
    df["rating_event"] = df["date"].dt.strftime("%m-%d-%y")
    df["rater_code"] = pd.Categorical(df["rater"]).codes + 1
    df["rating_event_code"] = pd.Categorical(df["rating_event"]).codes + 1
    
    # create data handler
    datahandler = utils.DataHandler(filepath=config["data_path"])
    datahandler.load_data(df=df)
    datahandler.preprocess_data()

    model_df = datahandler.model_data.copy()
    model_df["holdout_date"] = model_df["date"].dt.strftime("%Y-%m-%d")
    
    # train/test split: hold out ~20% of rating dates, approximately evenly spaced over the trial timeline
    held_out_dates = _load_held_out_dates(holdout_test_data_path)
    if held_out_dates is None:
        unique_dates = (
        model_df[["holdout_date", "date"]]
        .drop_duplicates()
        .sort_values(["date", "holdout_date"])
        .reset_index(drop=True)
        )
        num_dates = len(unique_dates)
        num_held_out = max(1, round(num_dates * 0.2))
        held_out_idx = np.round(np.linspace(1, num_dates, num_held_out + 2))[1:-1].astype(int)
        held_out_dates = unique_dates.iloc[held_out_idx]["holdout_date"].tolist()

    train_df = model_df[~model_df["holdout_date"].isin(held_out_dates)].copy()
    test_df = model_df[model_df["holdout_date"].isin(held_out_dates)].copy()
    # re-encode rating_event_code to be continuous after removing held-out events
    train_df["rating_event_code"] = pd.Categorical(train_df["rating_event"]).codes + 1
    datahandler.model_data = train_df
    logger.info(
        f"Train/test split: {len(train_df)} train rows, {len(test_df)} test rows "
        f"({len(held_out_dates)} held-out dates: {held_out_dates})."
    )
    
    datahandler.generate_stan_data(**config["stan_additional_data"])
    
    # re-encode held-out dates to be continuous and inject into stan_data
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
    split_metadata = {
        "held_out_dates": held_out_dates,
        "train_dates": sorted(train_df["holdout_date"].unique().tolist()),
        "test_dates": sorted(test_df["holdout_date"].unique().tolist()),
    }
    with open(os.path.join(working_dir, "held_out_dates.json"), "w", encoding="utf-8") as file:
        json.dump(split_metadata, file, indent=2)
    print("Train dates:", split_metadata["train_dates"])
    print("Test dates:", split_metadata["test_dates"])
    
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
    # Save withheld test data for future assessment
    with open(working_dir + "/test_data.pkl", 'wb') as file:
        pickle.dump(test_df, file)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the Stan model fitting process.")
    parser.add_argument("config_file", type=str, help=f"Configuration file name located in {CONFIG_DIR}")
    parser.add_argument("data_path", type=str, help=f"File to process")
    parser.add_argument("working_dir", type=str, help=f"Working directory for this run")
    parser.add_argument("model_output_file", type=str, help=f"Output pickle file")
    parser.add_argument(
        "--holdout-test-data",
        type=str,
        default=None,
        help="Optional pickle or CSV file whose date column defines the held-out dates to reuse.",
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(
        args.config_file,
        args.data_path,
        args.working_dir,
        args.model_output_file,
        holdout_test_data_path=args.holdout_test_data,
    )
    #to execute
    #python nteprsm/model_umn_with_split.py config/annual_seasonality_model_with_split.yml kb2017/nj2/quality.csv model_runs/cross_val_nj2_quality fit_seasonality_nj2_quality_cross_val.pkl
