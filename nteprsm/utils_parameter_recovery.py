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

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ExpSineSquared, ConstantKernel

import random
import pickle

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

class ParameterRecovery:
    def __init__():
      pass
    def rbf_kernel(self, sigma, lengthscale, x1, x2, y1, y2):
        """
        Computes the Radial Basis Function (RBF) kernel value for given points.

        Args:
            sigma (float): Scale of the kernel.
            lengthscale (float): Lengthscale of the kernel.
            x1, x2 (float): X-coordinates of the points.
            y1, y2 (float): Y-coordinates of the points.

        Returns:
            float: The computed RBF kernel value.
        """
      dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
      return sigma**2*np.exp(-1*dist/(2*lengthscale))
    def generate_rating_sample(self, theta, tau):
        """
        Generates a random rating sample based on the Rasch model probability.

        Args:
            theta (float): Quality parameter.
            tau (array of length 8): Rating thresholds parameter.

        Returns:
            int: A randomly generated rating outcome.
        """
      probabilities = [rsm_probability(i, theta, tau) for i in range(9)]
      return np.random.choice(len(probabilities), p=probabilities)

    # generates a sample plot effects using 2d RBF kernel Gaussian Process
    def generate_plot_effect(self, num_row, num_col, lengthscale_plot, sigma_plot, jitter = 10**(-10)):
        """
        Generates a sample plot effect using a 2D RBF kernel Gaussian Process.

        Args:
            num_row (int): Number of rows in the grid.
            num_col (int): Number of columns in the grid.
            lengthscale_plot (float): Lengthscale for the plot effect.
            sigma_plot (float): Scale for the plot effect.
            jitter (float): Jitter for numerical stability.

        Returns:
            numpy.ndarray: A 2D array representing the plot effect of shape (num_row, num_col)
        """
      # grid set up
      grid_size = num_row*num_col
      plot_col = [i%num_col for i in range(grid_size)]
      plot_row = [i//num_col for i in range(grid_size)]
      cov = np.zeros((grid_size,grid_size))
      for i in range(grid_size):
        for j in range(grid_size):
          if i == j: cov[i][j] = 1 - jitter
          else: cov[i][j] = self.rbf_kernel(sigma_plot, lengthscale_plot, plot_col[i], plot_col[j],plot_row[i], plot_row[j])
      sample = np.random.multivariate_normal(np.zeros(grid_size), cov, size=1).reshape(num_row, num_col)
      return sample

    def generate_time_effect(self, lengthscale_f, sigma_f, intercept = 0):
        """
        Generates a time effect of an entry using a Gaussian Process with an Periodic kernel.

        Args:
            lengthscale_f (float): Lengthscale for the time effect.
            sigma_f (float): Scale for the time effect.
            intercept (float) : Mean of the Gaussian Process regression
        Returns:
            numpy.ndarray: A 1D array representing the time effect.
        """
      X = np.linspace(0, 1, 365, endpoint=False).reshape(-1, 1)
      kernel = sigma_f**2 * ExpSineSquared(periodicity=1, length_scale=lengthscale_f)
      gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.0)
      sample = gpr.sample_y(X, n_samples=1).flatten() + intercept
      return sample

    # shuffle some entries in random locations but try to make them a bit spread out
    def generate_entry_locations(self, num_row, num_col, acceptable_distance):
    """
    Generates entry locations on a grid, ensuring they are spread out.

    Args:
        num_row (int): Number of rows in the grid.
        num_col (int): Number of columns in the grid.
        acceptable_distance (float): Minimum acceptable distance between entries. Gradually reduces if it fails to find a setup.

    Returns:
        (entry_locs, entry_dists) : A tuple containing the entry locations and their corresponding minimum distances of a pair of entries.
    """
      # helper function
      def calc_min_dist(entry_pos):
        d = float('inf')
        for j in range(3): d = min((plot_col[entry_pos[(j+1)%3]] - plot_col[entry_pos[j]])**2 + (plot_row[entry_pos[(j+1)%3]] - plot_row[entry_pos[j]])**2, d)
        return d
      grid_size = num_row * num_col
      plot_col = [i%num_col for i in range(grid_size)]
      plot_row = [i//num_col for i in range(grid_size)]
      entry_locs = [i for i in range(grid_size)]
      random.shuffle(entry_locs)
      entry_locs = [entry_locs[3*i:3*i+3] for i in range(grid_size//3)]
      entry_dists = []
      # shuffle entries until the min_dist is achieved
      for i in range(len(entry_locs)):
        entry_dists.append(np.sqrt(calc_min_dist(entry_locs[i])))
      num_entries_shuffled = 2
      trials = 0
      failed_to_generate = False
      while (min(entry_dists) < acceptable_distance):
        trials += 1
        if trials > 500:
          failed_to_generate = True
          break
        smallest_dists = sorted(range(len(entry_dists)), key = lambda x : entry_dists[x])[:num_entries_shuffled]
        reshuffle_pool = []
        for i in smallest_dists:
          for j in range(3): reshuffle_pool.append(entry_locs[i][j])
        random.shuffle(reshuffle_pool)
        counter = 0
        for i in smallest_dists:
          entry_locs[i] = reshuffle_pool[counter*3:counter*3+3]
          entry_dists[i] = np.sqrt(calc_min_dist(entry_locs[i]))
          counter += 1
      if failed_to_generate: return self.generate_entry_locations(num_row, num_col, acceptable_distance - 0.01)
      else: return entry_locs, entry_dists
    def plot_to_entry(self, entry_locs):
        """
        Converts plot locations to entry indices.

        Args:
            entry_locs (list): List of entry locations.

        Returns:
            list: A list where each plot location is mapped to its entry index.
        """
      arr = [0 for i in range(np.array(entry_locs).reshape(-1,1).max()+1)]
      for i in range(len(entry_locs)):
        for entry in entry_locs[i]:
          arr[entry] = i
      return arr
    def generate_rater_taus(self, base_mean = np.array([-2.1, -1.5, -0.9, -0.3, 0.3, 0.9, 1.5, 2.1])*1.1, std_individual_taus = 0.25, std_overall_severity = 0.35):
        """
        Generates rater-specific taus by adding individual and overall variations.

        Args:
            base_mean (numpy.ndarray): Base mean tau values.
            std_individual_taus (float): Standard deviation for individual tau variations.
            std_overall_severity (float): Standard deviation for overall severity variation.

        Returns:
            numpy.ndarray: Generated rater taus.
        """
      return np.random.normal(loc=np.array(base_mean), scale=0.25, size=8) + 0.35*np.random.normal(0,1)
    
    def generate_data(self, num_row = 18, num_col = 15,
                  lengthscale_f = 3, sigma_f = 0.4,
                  lengthscale_plot = 3, sigma_plot = 0.5, jitter = 10**(-10),
                  num_raters = 5, base_mean = np.array([-2.1, -1.5, -0.9, -0.3, 0.3, 0.9, 1.5, 2.1])*1.1,
                  std_individual_taus = 0.25, std_overall_severity = 0.35,
                  rating_event_days = [i for i in range(0,365,10)],
                  rating_event_raters = [random.randint(0,num_raters-1) for i in range(len(rating_event_days))]):
      """
        Generates synthetic data for the rating model.

        Args:
            num_row (int): Number of rows in the grid.
            num_col (int): Number of columns in the grid.
            lengthscale_f (float): Lengthscale for the time effect.
            sigma_f (float): Scale for the time effect.
            lengthscale_plot (float): Lengthscale for the plot effect.
            sigma_plot (float): Scale for the plot effect.
            num_raters (int): Number of raters.
            base_mean (numpy.ndarray): Base mean tau values.
            std_individual_taus (float): Standard deviation for individual tau variations.
            std_overall_severity (float): Standard deviation for overall severity variation.
            rating_event_days (list): Days on which ratings occur.
            rating_event_raters (list): Raters for each rating event.

        Returns:
            tuple: A DataFrame of generated data and a dictionary of model parameters.
        """
      # parameters derived from those manually input
      grid_size = num_row*num_col
      num_entry = grid_size//3
      plot_col = [i%num_col for i in range(grid_size)]
      plot_row = [i//num_col for i in range(grid_size)]

      # generate entry locations
      entry_locs, entry_dists = self.generate_entry_locations(num_row, num_col, 5.8)
      pte = self.plot_to_entry(entry_locs)

      # generate plot effect
      plot_effect = self.generate_plot_effect(num_row, num_col, lengthscale_plot, sigma_plot, jitter)

      # generate time effect
      time_effects = []
      for i in range(num_entry):
        time_effects.append(self.generate_time_effect(lengthscale_f, sigma_f, np.random.normal(loc=0, scale=0.5)))

      # generate rater data
      rater_taus = []
      for i in range(num_raters):
        rater_taus.append(self.generate_rater_taus(base_mean, std_individual_taus, std_overall_severity))

      # save model parameters
      data_params = {
          'num_row':num_row,
          'num_col':num_col,
          'lengthscale_f':lengthscale_f,
          'sigma_f':sigma_f,
          'lengthscale_plot':lengthscale_plot,
          'sigma_plot':sigma_plot,
          'num_raters':num_raters,
          'base_mean':base_mean,
          'std_individual_taus':std_individual_taus,
          'std_overall_severity':std_overall_severity,
          'rater_taus':rater_taus,
          'rating_event_days':rating_event_days,
          'rating_event_raters':rating_event_raters,
          'time_effects':time_effects,
          'plot_effect':plot_effect,
          'pte':pte,
      }
      df_lists = []
      # generate fake data
      for i in range(len(rating_event_days)):
        day = rating_event_days[i]
        rater = rating_event_raters[i]
        tau = rater_taus[rater]

        for j in range(grid_size):
          plot_id = j
          entry = pte[j]
          theta = time_effects[entry][day] + plot_effect[plot_row[j]][plot_col[j]]
          quality = self.generate_rating_sample(theta, tau)
          # new_data = {'plot_id':j,
          #             'plot_col':plot_col[j],
          #             'plot_row':plot_row[j],
          #             'entry':pte[j],
          #             'day': rating_event_days[i],
          #             'rater': rater,
          #             'quality':quality}
          df_lists.append([j,plot_col[j],plot_row[j],entry,day,rater,quality])
      df = pd.DataFrame(df_lists, columns=['plot_id', 'plot_col', 'plot_row', 'entry','day', 'rater', 'quality'])
      stan_data = {
            "y": df['quality'].values,
            "N": len(df),
            "num_raters": num_raters,
            "num_entries": grid_size//3,
            "num_plots": grid_size,
            "num_categories": 9,
            "rater_id": df['rater']+1,
            "entry_id": df['entry'] + 1,
            "plot_id": df['plot_id'] + 1,
            "num_ratings_per_entry": len(rating_event_days)//(grid_size//3),
            "num_rows": num_rows,
            "num_cols": num_cols,
            "plot_row": df['plot_row'],
            "plot_col": df['plot_col'],
            "time": df['day']/365,
            "entry_cumcount": (df.groupby("entry").cumcount() + 1).values,
        }
      return stan_data, data_params, df