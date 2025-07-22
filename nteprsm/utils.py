import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from random import sample
from typing import Dict, Optional
import pytensor.tensor as pt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yaml
import warnings
from cmdstanpy.stanfit import CmdStanMCMC
from scipy.special import softmax
from scipy.spatial.distance import pdist, squareform

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


def rsm(theta, beta, tau):
    """
    Custom function to calculate probabilities based on theta, beta, and tau.
    """
    theta_minus_beta = theta - beta
    unsummed = pt.concatenate([pt.zeros((theta.shape[0], 1)), theta_minus_beta[:, None] - tau], axis=1)
    cumsum = pt.cumsum(unsummed, axis=1)
    exp_cumsum = pt.exp(cumsum - pt.max(cumsum, axis=1, keepdims=True))
    probs = exp_cumsum / pt.sum(exp_cumsum, axis=1, keepdims=True)
    return probs

def get_nuts_kwargs(config):
    """
    Extract NUTS sampler parameters from the configuration.
    """
    return {
        'target_accept': config['sampling'].get('adapt_delta'),
        'max_treedepth': config['sampling'].get('max_treedepth'),
    }

def get_sample_kwargs(config, nuts_kwargs):
    """
    Extract sampling parameters from the configuration.
    """
    return {
        'draws': config['sampling'].get('iter_sampling'),
        'tune': config['sampling'].get('iter_warmup'),
        'chains': config['sampling'].get('parallel_chains'),
        'return_inferencedata': True,
        'random_seed': config['sampling'].get('seed'),
        'progressbar': config['sampling'].get('show_progress'),
        'init': 'adapt_diag',
        'nuts': nuts_kwargs,
    }

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
    """
    DataHandler is a utility class for managing NTEP's turfgrass evaluation data,
    supporting a structured pipeline from raw CSV input to Stan-compatible data
    dictionaries for Bayesian modeling.

    It operates in three stages:
    1. `raw_data`: Unmodified data loaded from a CSV or DataFrame.
    2. `model_data`: Transformed and enriched version used for diagnostics and modeling.
    3. `stan_data`: A minimal, structured dictionary for use with Stan models.

    Attributes:
        target (str): The name of the target variable column.
        filepath (Path or None): Path to the input CSV file (if provided).
        logger (Logger): Logger instance for messaging and warnings.
        raw_data (pd.DataFrame or None): Untransformed input data.
        model_data (pd.DataFrame or None): Transformed modeling data.
        stan_data (dict or None): Stan-compatible data dictionary.
    """
    def __init__(
        self,
        target: str = "value",
        filepath: Optional[Path | str] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initializes the DataHandler.

        Args:
            target (str): The name of the target variable to model. Defaults to "value".
            filepath (str or Path, optional): Path to a CSV file to automatically load. Defaults to None.
            logger (Logger, optional): Optional logger to use. If not provided, a default logger is set up.
        """
        self.logger = logger or setup_logging(LOG_DIR)
        self.target = target
        self.filepath = Path(filepath) if filepath else None
        self.raw_data: Optional[pd.DataFrame]= None
        self.model_data: Optional[pd.DataFrame] = None
        self.stan_data: Optional[Dict]= None

        # Automatically load data if filepath is provided
        if self.filepath:
            if self.filepath.suffix.lower() != ".csv":
                self.logger.warning(f"Unsupported file type: {self.filepath.suffix}")
            else:
                try:
                    self.load_data(pd.read_csv(self.filepath))
                except Exception as e:
                    self.logger.error(f"Failed to load data: {e}")

    def __repr__(self):
        """
        Returns a string representation of the DataHandler instance.

        Returns:
            str: Representation showing the target variable and file path (if any).
        """
        return f"<DataHandler target={self.target} file={self.filepath}>"

    def load_data(self, df: pd.DataFrame) -> None:
        """
        Loads and stores a raw input DataFrame as the baseline dataset (`raw_data`).

        This method stores the original data without any modification. It is
        intended to be the immutable source of truth for later transformations.

        Args:
            df (pd.DataFrame): Input data containing raw rating records.
        """
        self.raw_data = df.copy(deep=True)  # Immutable ground truth
        self.logger.info("Raw data successfully loaded.")

    def preprocess_data(self) -> None:
        """
        Processes the raw data to generate `model_data` for modeling.

        Operations include:
        - Parsing and adjusting dates for leap years.
        - Encoding categorical columns (`rater`, `rating_event`) into numeric codes.
        - Normalizing the target variable to start from 0.
        - Calculating adjusted time of year and cumulative entry counts.
        - Ensuring categorical code columns are continuous from 1 to max.

        Raises:
            ValueError: If `raw_data` is not loaded or required columns are missing.
        """
        if self.raw_data is None:
            raise ValueError("Call `load_data()` before preprocessing.")

        df = self.raw_data.copy()

        required = ["rater", "date", "entry_code", "plot_code", "row", "col", self.target]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Process date
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if df["date"].isna().any():
            warnings.warn(f"Some dates could not be parsed: {df[df['date'].isna()].index.tolist()}")

        # Derived features
        df["rating_event"] = df["rater"].astype(str) + "-" + df["date"].dt.strftime("%m-%d-%y")
        df["rating_event_code"] = pd.Categorical(df["rating_event"]).codes + 1
        df["rater_code"] = pd.Categorical(df["rater"]).codes + 1

        # Time features (adjust for leap years)
        day_of_year = df["date"].dt.day_of_year
        is_leap = df["date"].dt.is_leap_year
        leap_correction = ((is_leap) & (day_of_year >= 60)).astype(int)
        df["adj_day_of_year"] = day_of_year - leap_correction
        df["adj_time_of_year"] = df["adj_day_of_year"] / 365.0

        # Target normalization
        df[self.target] = df[self.target] - df[self.target].min()


        # Ensure categorical codes are continuous
        check_columns = [
            col for col in df.columns
            if col.endswith("_code") and col != self.target
        ]
        for col in check_columns:
            codes = df[col].unique()
            if not np.array_equal(np.sort(codes), np.arange(1, codes.max() + 1)):
                warnings.warn(f"{col} not continuous. Re-encoding.")
                df[col] = pd.Categorical(df[col]).codes + 1

        self.model_data = df
        self.logger.info("Preprocessing complete. Model data ready.")

    def generate_stan_data(self, plot_data: Optional[pd.DataFrame] = None, **kwargs) -> None:
        """
        Generates a Stan-compatible data dictionary (`stan_data`) from `model_data`.

        The dictionary contains structured arrays and scalars used in Stan models,
        including codes, dimensions, target vector, and optional plot spatial data.

        Args:
            plot_data (pd.DataFrame, optional): External plot layout (mean row/col per plot_code).
                If not provided, uses group-level means from `model_data`.

            **kwargs: Additional key-value pairs to inject into `stan_data`.

        Raises:
            ValueError: If `model_data` is not prepared.
        """
        if self.model_data is None:
            raise ValueError("Call `preprocess_data()` before generating Stan data.")

        df = self.model_data

        if plot_data is None:
            plot_data = df.groupby("plot_code")[["row", "col"]].mean()

        self.stan_data = {
            "y": df[self.target].values,
            "N": len(df),
            "num_ratings": int(df["rating_event_code"].max()),
            "num_raters": int(df["rater_code"].max()),
            "num_entries": int(df["entry_code"].max()),
            "num_plots": int(df["plot_code"].max()),
            "y_max": int(df[self.target].max()),
            "rating_event_code": df["rating_event_code"].values,
            "entry_code": df["entry_code"].values,
            "plot_code": df["plot_code"].values,
            "rater_code": df["rater_code"].values,
            "DIST": self._calculate_distance_matrix(),
            "num_rows": int(plot_data["row"].max()),
            "num_cols": int(plot_data["col"].max()),
            "plot_row": plot_data["row"].astype(int).values,
            "plot_col": plot_data["col"].astype(int).values,
            "time": df[["rating_event_code", "adj_time_of_year"]]
                        .drop_duplicates()
                        .sort_values("rating_event_code")
                        .set_index("rating_event_code")
                        .values.reshape(-1)
        }

        self.stan_data.update(kwargs)
        self.logger.info("Stan data dictionary created.")

    def _calculate_distance_matrix(self) -> np.ndarray:
        """
        Computes the pairwise Euclidean distance matrix between plots using row/col coordinates.

        Extracts unique plot positions from `model_data` and computes the
        distance matrix using scipy's pdist and squareform utilities.

        Returns:
            np.ndarray: A square matrix of pairwise distances between plots.

        Raises:
            ValueError: If `model_data` is missing or required coordinate columns are absent.
        """
        if self.model_data is None:
            raise ValueError("Model data not available.")

        required = {"plot_code", "row", "col"}
        if not required.issubset(self.model_data.columns):
            raise ValueError("Missing plot coordinates for distance matrix.")

        coords = (
            self.model_data[["plot_code", "row", "col"]]
            .drop_duplicates()
            .sort_values("plot_code")[["row", "col"]]
            .values
        )

        return squareform(pdist(coords))


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

    def get_predicted_statistics(self, func, *args, **kwargs):
        """
        Get model predictions as a dataframe, with pred_day as an additional column.

        Args:
            func (callable): The function to apply to the predictions.
            *args: Additional positional arguments to pass to the function.
            **kwargs: Additional keyword arguments to pass to the function.

        Returns:
            pd.DataFrame: A dataframe containing the model predictions and adj_day_of_year.
        """
        num_preds = self.datahandler.stan_data["pred_N"]
        data = func(self.stanmcmc.pred_time_effect, axis=0, *args, **kwargs).T
        pred_data = pd.DataFrame(data)
        pred_data = pred_data.assign(adj_day_of_year=
                        np.array(range(1, num_preds + 1)) / num_preds * 365)
        pred_data.set_index("adj_day_of_year", inplace=True)
        pred_data.columns.name = "entry_name_code"
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
            pred_means.index, bins=MONTH_BINS, labels=MONTH_ABBR, right=False
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
                entry_codes, key=lambda e: monthly_means.mean(axis=1).loc[e], 
                reverse=True
            )
        elif sort_by == sort_by and sort_by in MONTH_ABBR:
            entry_codes = sorted(
                entry_codes, key=lambda e: monthly_means.loc[e, sort_by],
                reverse=True
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
            x_pred = means.index
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
            # title="Mean Time Effect",
            xaxis=dict(
                tickmode="array",
                tickvals=MONTH_BINS[:-1],
                ticktext=MONTH_ABBR,
                tickfont=dict(size=18),
            ),
            yaxis_title="Predicted Seasonality in Turf Quality",
            yaxis=dict(title_font=dict(size=20)),
            legend=dict(font=dict(size=16)),  # Increase the font size for the legend
            # title_font=dict(size=24),
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
