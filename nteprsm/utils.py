import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from random import sample
from typing import Dict, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yaml
from cmdstanpy.stanfit import CmdStanMCMC
from scipy.special import softmax

from nteprsm.constants import MONTH_ABBR, MONTH_BINS
from settings import LOG_DIR


def load_config(path_to_config: str) -> dict:
    """
    load a configuration file

    Args:
        path_to_config (str): a file path to the YML file with model configuration

    Returns:
        dict: a dictionary of configuration
    """
    with open(path_to_config, "r", encoding="UTF-8") as stream:
        return yaml.safe_load(stream)


def setup_logging(log_directory="logs"):
    """
    Configures and sets up the centralized logging for the application,
    directing log output to a specified directory. Each log file created will
    have a unique name based on the datetime when the application was run.

    Args:
        log_directory (str): The directory where log files will be stored.

    Returns:
        logging.Logger: The configured logger object.

    Creates:
        A log file in the specified directory with a unique name that includes
        the current date and time.
    """
    # Ensure the log directory exists
    os.makedirs(log_directory, exist_ok=True)

    # Generate a log file name that includes the current date and time
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"NtepRsm_{current_time}.log"
    log_path = os.path.join(log_directory, log_filename)

    # Get or create a logger
    logger = logging.getLogger("NtepRsm")

    # Check if the logger already has handlers configured
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)  # Set the base logging level

        # Create handlers for both file and console
        file_handler = RotatingFileHandler(
            log_path, maxBytes=1024 * 1024 * 5, backupCount=5
        )
        console_handler = logging.StreamHandler()

        # Create a formatter and add it to the handlers
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        # Prevent logging from propagating to the root logger
        logger.propagate = False
    else:
        # Log a message indicating that logging is already configured
        logger.debug("Logging is already configured.")

    return logger


def rsm_probability(y, theta, tau):
    """
    Calculates the probability of a given class label in the model.

    Args:
    y (int): The class label for which the probability is calculated.
    theta (np.ndarray): An array of model parameters.
    tau (np.ndarry): The threshold parameters for the model.

    Returns:
    float: The probability of the given class label.

    """
    unsummed = np.concatenate(([0], theta - tau))
    probs = softmax(np.cumsum(unsummed))
    return probs[y]


class DataHandler:
    def __init__(
        self,
        filepath: Optional[Path | str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the DataHandler with an optional file path for CSV data.

        Args:
            filepath (Path, optional): The path to the file to load, only accept
            CSV files for now. Defaults to None.
            logger (Optional[logging.Logger]): Logger for logging data handling
            processes.
        """
        self.filepath = Path(filepath)
        self.logger = logger if logger is not None else setup_logging(LOG_DIR)
        self.raw_data = None
        self.model_data = None
        self.stan_data = None

    def load_data(self):
        """
        Load data from file specified during class initialization.

        Raises:
            FileNotFoundError: If the CSV file is not found at the specified path.
            ValueError: If the file extension is not CSV.
        """
        if not self.filepath:
            self.logger.error("File path is not provided.")
            raise ValueError("File path must be provided.")
        if self.filepath.suffix == ".csv":
            self.logger.info(f"Loading data from {self.filepath}...")
            self.raw_data = pd.read_csv(self.filepath)
        else:
            raise FileNotFoundError(
                f"File {self.filepath} not found or is not a CSV file."
            )

    def preprocess_data(self):
        """
        Preprocess the loaded data. This method assumes data is already loaded
        into `self.raw_data`.

        - Convert all column names to lowercase.
        - Encode categorical variables.
        """
        if self.raw_data is None:
            raise Exception("Data not loaded. Call 'load_data' first.")

        self.logger.info("Start preprocessing data...")
        # make sure all column names to lower case
        model_data = self.raw_data.copy()
        model_data.columns = [col.lower() for col in model_data.columns]

        model_data = model_data.assign(
            entry_name_code=pd.Categorical(model_data["entry_name"]).codes,
            plt_id_code=pd.Categorical(model_data["plt_id"]).codes,
            rater_code=pd.Categorical(model_data["rater"]).codes,
            rating_event_code=pd.Categorical(model_data["rating_event"]).codes,
            date=pd.to_datetime(model_data["date"]),
        )
        # adjust day of year for leap year, Feb 19th is the 60th day of the year
        # day_of_year on or past Feb 29th in a leap year will reduced by 1.
        model_data["adj_day_of_year"] = model_data.date.dt.day_of_year - (
            model_data.date.dt.is_leap_year * model_data.date.dt.day_of_year >= 60
        )
        model_data["adj_time_of_year"] = model_data.adj_day_of_year / 365
        model_data["entry_cumcount"] = model_data.groupby("entry_name").cumcount() + 1
        self.model_data = model_data
        self.logger.info("Data preprocessing completed.")

    def get_processed_data(self) -> pd.DataFrame:
        """
        Return the processed data.

        Returns:
            pd.DataFrame: The processed data.
        """
        if self.processed_data is None:
            raise Exception(
                "Data has not been processed. Call 'preprocess_data' first."
            )
        return self.processed_data

    def generate_stan_data(
        self,
        plot_data: Optional[pd.DataFrame] = None,
        target: str = "quality",
        **kwargs,
    ) -> None:
        """
        Generate a data block formatted for input into a Stan model. This method
        prepares the data by structuring it into a dictionary with keys and
        values that Stan models require.

        Args:
            plot_data (Optional[pd.DataFrame]): Data containing plot layout
                information including 'row' and 'col' which can optionally be
                provided externally. If None, it defaults to using the 'row' and
                'col' values from `self.model_data`.
            target (str): The name of the target variable column in
                `self.model_data` to use for the Stan model. Defaults to
                'quality'.

        Raises:
            ValueError: If `self.model_data` is None, indicating that data has
                not been loaded or processed properly before this method is called.
        """
        if self.model_data is None:
            raise ValueError(
                "Model data has not been loaded or preprocessed. \
                Please load and preprocess data before generating Stan data."
            )

        self.logger.info("Generating data dictionary for Stan...")
        if plot_data is None:
            plot_data = self.model_data.groupby("plt_id_code")[["row", "col"]].mean()

        stan_data = {
            "y": (self.model_data[target] - self.model_data[target].min()).values,
            "N": len(self.model_data[target]),
            "num_raters": int(self.model_data.rater.nunique()),
            "num_entries": int(self.model_data.entry_name.nunique()),
            "num_plots": int(self.model_data.plt_id.nunique()),
            "num_categories": int(self.model_data[target].nunique()),
            "rater_id": self.model_data.rater_code.values + 1,
            "entry_id": self.model_data.entry_name_code.values + 1,
            "plot_id": self.model_data.plt_id_code.values + 1,
            "num_ratings_per_entry": self.model_data.groupby("entry_name")
            .count()["plt_id"]
            .max(),
            "num_rows": int(plot_data.row.max()),
            "num_cols": int(plot_data.col.max()),
            "plot_row": plot_data.row.astype(int).values,
            "plot_col": plot_data.col.astype(int).values,
            "time": self.model_data.adj_time_of_year.values,
            "entry_cumcount": self.model_data.entry_cumcount.values,
        }

        stan_data.update(kwargs)
        self.stan_data = stan_data

    def get_stan_data(self) -> Dict:
        """
        Return the processed Stan data.

        Returns:
            Dict: The processed Stan data.
        """
        if self.stan_data is None:
            self.logger.error(
                "Stan data has not been generated. \
                Call 'generate_stan_data' first."
            )
            raise Exception(
                "Stan data has not been generated. \
                Call 'generate_stan_data' first."
            )
        return self.stan_data

    def map_name2code(self, column_name, code_column_name, invert=False):
        """
        Retrieves a dictionary mapping names to codes from specified columns.

        Args:
            column_name(str): The name of the column containing the names(
            e.g., 'ENTRY_NAME', 'RATER').
            code_column_name(str): The name of the column containing the codes
            (e.g., 'ENTRY_NAME_CODE', 'RATER_CODE').
            invert(bool): If True, returns a dictionary mapping codes to names.

        Returns:
            dict: Depending on 'invert', returns either a dict of {name: code}
            or {code: name}.
        """
        name2code = dict(self.model_data.groupby(column_name)[code_column_name].first())

        if invert:
            # Using dictionary comprehension for inverting the dictionary
            return {code: name for name, code in name2code.items()}

        return name2code


class PosteriorSampleAnalysis:
    def __init__(
        self,
        datahandler: DataHandler,
        stanmcmc: CmdStanMCMC,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initializes the PosteriorSampleAnalysis class with instances of
        DataHandler and CmdStanMCMC, along with setting up a logger for the class.

        Args:
            data_handler (DataHandler): An instance of the DataHandler class for
            managing and preprocessing data.
            stan_mcmc (CmdStanMCMC): An instance of CmdStanMCMC containing the
            results of a Stan model fit.
            logger (Optional[logging.Logger]): Logger for logging data handling
            processes.
        """
        self.datahandler = datahandler
        self.stanmcmc = stanmcmc
        self.logger = logger if logger is not None else setup_logging(LOG_DIR)

    def get_predicted_statistics(self, func, *arg):
        """get model predictions as a dataframe, with pred_day as an additional
        column
        """
        num_preds = self.datahandler.stan_data["pred_N"]
        pred_day = np.array(range(1, num_preds + 1)) / num_preds * 365
        data = func(self.stanmcmc.pred_time_effect, axis=0, *arg).T
        pred_data = pd.DataFrame(data)
        pred_data = pred_data.assign(pred_day=pred_day)
        return pred_data

    def get_predicted_monthly_means(
        self,
        pred_means,
    ) -> pd.DataFrame:
        """
        Compute the monthly mean values of entries from the output of a fitted
        Stan model, specifically from a CmdStanMCMC object. The function
        processes the prediction effects stored in `pred_time_effect` in the
        Stan model output, calculates mean values for each entry across monthly
        intervals, and organizes them into a readable format.

        Returns:
            pd.DataFrame: A DataFrame where rows corresponds to entry names, and
            columns corresponds to a month (from 'Jan' to 'Dec'), and values
            represent the average time effects for that entry across the
            respective month. We also added a column for entry name.
        """
        pred_means["month"] = pd.cut(
            pred_means["pred_day"], bins=MONTH_BINS, labels=MONTH_ABBR, right=False
        )
        monthly_means = (
            pred_means.groupby("month")[
                list(range(self.datahandler.get_stan_data()["num_entries"]))
            ]
            .mean()
            .T
        )
        return monthly_means

    def plot_time_effect(
        self,
        entries=26,  # can be an int or a list of entry identifiers
        colors=px.colors.qualitative.Dark24,  # Plotly colors
        ci=None,  # None or a float
        sort_by="annual",  # Options: 'annual' or 'month'
        dimensions=None,  # Optional: None or a tuple (width, height)
    ):
        """
        Plots the time effect of entries based on model predictions with optional sorting and credit intervals.

        Args:
            entries (int or list): Either the number of entries to randomly select or a specific list of entries.
            colors (list): List of colors for plotting.
            credit_interval (float, optional): Confidence interval to display (e.g., 0.95 for 95% CI).
            sort_entries (str): Method to sort the entries; defaults to 'weighted'.
            dimensions (tuple, optional): Dimensions of the plot as (width, height).

        Returns:
            A Plotly figure object containing the plotted time effect.
        """
        # Handle entry input types and determine selection of entries
        name2code = self.datahandler.map_name2code("entry_name", "entry_name_code")
        if isinstance(entries, list):
            entry_codes = []
            for entry in entries:
                if isinstance(entry, int) and (
                    0 <= entry < self.datahandler.stan_data["num_entries"]
                ):
                    entry_codes.append(entry)
                elif isinstance(entry, str) and entry in name2code:
                    entry_codes.append(name2code[entry])
                else:
                    self.logger.warningn(f"Skip invalid entry {entry}!")
            entry_codes = list(set(entry_codes))  # Remove duplicates

        elif isinstance(entries, str) and entries.lower() == "all":
            entry_codes = list(range(self.datahandler.stan_data["num_entries"]))
            self.logger.info("Plotting all entries...")

        elif isinstance(entries, int):
            entry_codes = sample(
                range(self.datahandler.stan_data["num_entries"]), entries
            )

        # Retrieve and prepare data
        means = self.get_predicted_statistics(np.mean)
        # Sorting entries if required
        monthly_means = self.get_predicted_monthly_means(means)

        if sort_by == "annual":
            entry_codes = sorted(
                entry_codes, key=lambda e: monthly_means.mean(axis=1).loc[e]
            )
        elif sort_by == sort_by and sort_by in MONTH_ABBR:
            entry_codes = sorted(
                entry_codes, key=lambda e: monthly_means.loc[e, sort_by]
            )
        else:
            raise ValueError(
                f"{sort_by} is an invalid value. Currently only "
                + f"accept one {MONTH_ABBR}."
            )

        # Retrieve rating data
        ratings = self.datahandler.model_data
        ratings = ratings.loc[
            ratings.entry_name_code.isin(entry_codes),
            ["adj_day_of_year", "entry_name_code"],
        ]
        code2name = self.datahandler.map_name2code(
            "entry_name", "entry_name_code", invert=True
        )
        if ci is not None and 0 < ci < 1:
            y_lbs = self.get_predicted_statistics(np.quantile, 0.5 * (1 - ci))
            y_ubs = self.get_predicted_statistics(np.quantile, 0.5 * (1 + ci))
        # Set up Plotly graph
        fig = go.Figure()

        # prepare variables and plot
        for ix, code in enumerate(entry_codes):
            # retrieve fitted values
            x = ratings.loc[ratings.entry_name_code == code, "adj_day_of_year"].values
            idx = np.argsort(x)
            x_fitted = x[idx]
            y_fitted = self.stanmcmc.time_effect[:, code].mean(axis=0)[idx]
            entry_name = code2name[code]
            # plotting fitted values
            fig.add_trace(
                go.Scatter(
                    x=x_fitted,
                    y=y_fitted,
                    mode="markers",
                    marker=dict(size=5, color=colors[ix % len(colors)]),
                    name=entry_name,
                    legendgroup=entry_name,
                )
            )
            # plot predicted values for the whole year
            x_pred = means.pred_day.values
            y_pred = means[code]
            fig.add_trace(
                go.Scatter(
                    x=x_pred,
                    y=y_pred,
                    mode="lines",
                    line=dict(width=1.5, color=colors[ix % len(colors)]),
                    name=entry_name,
                    legendgroup=entry_name,
                    showlegend=False,
                    hoverinfo="none",
                )
            )
            if ci is not None and 0 < ci < 1:
                y_lb = y_lbs[code]
                y_ub = y_ubs[code]
                fig.add_trace(
                    go.Scatter(
                        x=x_pred,
                        y=y_lb,
                        mode="lines",
                        line=dict(width=0.5, color=colors[ix % len(colors)]),
                        name=entry_name,
                        legendgroup=entry_name,
                        showlegend=False,
                        hoverinfo="none",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=x_pred,
                        y=y_ub,
                        mode="lines",
                        line=dict(width=0.5, color=colors[ix % len(colors)]),
                        name=entry_name,
                        legendgroup=entry_name,
                        showlegend=False,
                        fill="tonexty",
                        hoverinfo="none",
                    )
                )
        fig.update_layout(
            title="Mean Time Effect",
            xaxis=dict(
                tickmode="array",
                tickvals=MONTH_BINS[:-1],
                ticktext=MONTH_ABBR,
                tickfont=dict(size=18),
            ),
            yaxis_title="Effect",
            yaxis=dict(title_font=dict(size=20)),
            legend=dict(font=dict(size=16)),  # Increase the font size for the legend
            title_font=dict(size=24),
        )
        if dimensions:
            fig.update_layout(width=dimensions[0], height=dimensions[1])
        return fig

    def plot_rater_characteristic_curve(
        self,
        rater,
        min_theta=-6,
        max_theta=6,
        resolution=500,
        colors=px.colors.diverging.Spectral,
        dimensions=None,
    ) -> go.Figure:
        """
        Plot the characteristic curves for raters based on the fitted Stan model.

        Args:
            rater_id (int, optional): The rater ID to plot. If None, all raters
                will be plotted. Defaults to None.
            dimensions (tuple, optional): Dimensions of the plot as (width, height).

        Returns:
            A Plotly figure object containing the plotted characteristic curves.
        """
        # handle rater input types
        rater2code = self.datahandler.map_name2code("rater", "rater_code")
        if isinstance(rater, str) and rater in rater2code:
            code = rater2code[rater]
        elif isinstance(rater, int) and (
            0 <= rater < self.datahandler.stan_data["num_raters"]
        ):
            code = rater
        else:
            raise ValueError(f"Invalid rater {rater}!")
        # retrieve and prepare data
        taus = self.stanmcmc.tau_rater[:, code, :].mean(axis=0)
        taus_with_bounds = np.concatenate(([min_theta], taus, [max_theta]))
        x = np.linspace(min_theta, max_theta, int((max_theta - min_theta) * resolution))
        num_categories = self.datahandler.stan_data["num_categories"]
        fig = go.Figure()
        for i in range(num_categories):
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=[rsm_probability(i, theta, taus) for theta in x],
                    line=dict(width=2, color=colors[i]),
                    name=str(i + 1),
                )
            )
            fig.add_shape(
                type="rect",
                x0=taus_with_bounds[i],
                x1=taus_with_bounds[i + 1],
                y0=1.02,
                y1=1.1,
                fillcolor=colors[i],
            )
            if i != num_categories - 1:
                fig.add_shape(
                    type="line",
                    x0=taus[i],
                    x1=taus[i],
                    y0=0,
                    y1=1,
                    line=dict(color=colors[i], dash="dot"),
                )

        fig.update_layout(
            xaxis_title="Turf Quality on Latent Scale",
            yaxis_title="Probability",
            legend=dict(x=1.02, y=1),
        )
        if dimensions:
            fig.update_layout(width=dimensions[0], height=dimensions[1])
        return fig
