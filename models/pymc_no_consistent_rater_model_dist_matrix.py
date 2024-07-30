import pymc as pm
import numpy as np
import arviz as az
import pandas as pd
import pytensor.tensor as pt
from pathlib import Path
from nteprsm import utils
from settings import CONFIG_DIR, LOG_DIR
from datetime import datetime

# Setup logging
logger = utils.setup_logging(LOG_DIR)

def load_and_preprocess_data(config_file):
    """
    Load and preprocess data based on the given configuration file.
    """
    config = utils.load_config(config_file)
    datahandler = utils.DataHandler(filepath=config["data_path"], logger=logger)
    datahandler.load_data()
    datahandler.preprocess_data()

    # Load temperature data
    temp_data = pd.read_csv(config["temperature_path"], sep=';')
    temp_data['DATE'] = pd.to_datetime(temp_data['DATE'])

    # Ensure the date column in the main data is in the correct format
    datahandler.model_data['date'] = pd.to_datetime(datahandler.model_data['date'])

    # Merge temperature data with main data
    datahandler.model_data = pd.merge(datahandler.model_data, temp_data, left_on='date', right_on='DATE', how='left')

    # Set the index to 'date' for interpolation
    datahandler.model_data.set_index('date', inplace=True)

    # Interpolate missing temperature values if any
    datahandler.model_data['TEMPERATURE'] = datahandler.model_data['TEMPERATURE'].interpolate(method='time')

    # Reset index back to original
    datahandler.model_data.reset_index(inplace=True)

    additional_data = config.get("additional_data", {})
    datahandler.generate_stan_data(**additional_data)

    # Add temperature data to stan_data
    datahandler.stan_data['temperature'] = datahandler.model_data['TEMPERATURE'].values
    datahandler.stan_data['dates'] = (datahandler.model_data['date'] - datahandler.model_data['date'].min()).dt.days.values

    return datahandler.stan_data, config

def growth_potential(temp, is_cool_season=True):
    """
    Calculate growth potential based on temperature.
    """
    t_opt = 20 if is_cool_season else 31
    var = 5.5 if is_cool_season else 7
    return np.exp(-0.5 * ((temp - t_opt) / var) ** 2)

def create_pymc_model(data):
    """
    Create a PyMC model based on the preprocessed data, including temperature effects.
    """
    with pm.Model() as model:
        # Define priors
        beta_free = pm.Normal("beta_free", mu=0, sigma=2, shape=data['num_raters']-1)
        tau_free = pm.Normal("tau_free", mu=0, sigma=2, shape=data['num_categories']-1)
        sigma_e_plot = pm.HalfStudentT("sigma_e_plot", nu=3, sigma=1)
        alpha_plot = pm.HalfStudentT("alpha_plot", nu=3, sigma=1)
        length_scale_plot = pm.Gamma("length_scale_plot", alpha=5, beta=5)
        sigma_e_temp = pm.HalfStudentT("sigma_e_temp", nu=3, sigma=1)
        alpha_temp = pm.HalfStudentT("alpha_temp", nu=3, sigma=1)
        length_scale_temp = pm.Gamma("length_scale_temp", alpha=5, beta=5)

        # Transform parameters
        beta = pm.Deterministic("beta", pt.concatenate([beta_free, -pt.sum(beta_free, keepdims=True)]))
        tau = pm.Deterministic("tau", pt.concatenate([tau_free, -pt.sum(tau_free, keepdims=True)]))

        # Define Gaussian Process for plot effects using the distance matrix
        DIST = pt.as_tensor_variable(data['DIST'])
        cov_plot = alpha_plot**2 * pt.exp(-0.5 * (DIST * length_scale_plot)**2) + pt.eye(data['num_plots']) * sigma_e_plot**2
        plot_effect = pm.MvNormal("plot_effect", mu=0, cov=cov_plot, shape=data['num_plots'])

        # Calculate growth potential (GP)
        temp_mean = growth_potential(data['temperature'])
        
        # Define Gaussian Process for temperature effects using date distance
        dates = pt.as_tensor_variable(data['dates'])
        num_dates = dates.shape[0]
        date_distances = pt.abs(dates[:, None] - dates[None, :])
        cov_temp = alpha_temp**2 * pt.exp(-0.5 * (date_distances / length_scale_temp)**2) + pt.eye(num_dates) * sigma_e_temp**2
        
        temp_effects = pm.MvNormal("temp_effects", mu=temp_mean, cov=cov_temp, shape=(data['num_entries'], data['num_ratings_per_entry']))

        # Adjusted turf quality
        theta = pm.Deterministic("theta", plot_effect[data['plot_id']-1] + temp_effects[data['entry_id']-1, data['plot_id']-1])

        # Likelihood
        probs = utils.rsm(theta, beta[data['rater_id']-1], tau)
        y = pm.Categorical("y", p=probs, observed=data['y'])

    return model

def run_model(model, config):
    """
    Run the PyMC model using the NUTS sampler.
    """
    with model:
        # Get NUTS and sampling parameters from config
        nuts_kwargs = utils.get_nuts_kwargs(config)
        sample_kwargs = utils.get_sample_kwargs(config, nuts_kwargs)
        # Sample from the model
        trace = pm.sample(**sample_kwargs)
    return trace

def extract_samples(trace):
    """
    Extract relevant samples from the trace.
    """
    plot_effect_samples = trace.posterior['plot_effect'].values
    beta_samples = trace.posterior['beta'].values
    tau_samples = trace.posterior['tau'].values
    temp_effect_samples = trace.posterior['temp_effects'].values

    return plot_effect_samples, beta_samples, tau_samples, temp_effect_samples

def calculate_probs(theta, beta, tau):
    """
    Calculate probabilities for a single set of parameters.
    """
    unsummed = np.concatenate([np.zeros(1), theta - beta - tau])
    cumsum = np.cumsum(unsummed)
    exp_cumsum = np.exp(cumsum - np.max(cumsum))
    return exp_cumsum / np.sum(exp_cumsum)

def main(config_file):
    """
    Main function to load data, create the model, run it, and analyze the results.
    """
    # Load and preprocess data
    data, config = load_and_preprocess_data(config_file)
    # Create PyMC model
    model = create_pymc_model(data)
    # Run the model
    trace = run_model(model, config)

    # Get summary statistics
    summary = az.summary(trace)
    logger.info("Summary Statistics:")
    logger.info(summary)

    # Save the trace to a NetCDF file
    output_dir = Path(config['sampling']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    netcdf_filename = output_dir / f"trace_{timestamp}.netcdf"
    az.to_netcdf(trace, filename=netcdf_filename)
    logger.info(f"Full trace saved to {netcdf_filename}")

    # Extract samples
    plot_effect_samples, beta_samples, tau_samples, temp_effect_samples = extract_samples(trace)

    all_probs = []
    for n in range(len(data['entry_id'])):
        entry_index = data['entry_id'][n] - 1
        plot_index = data['plot_id'][n] - 1
        rater_index = data['rater_id'][n] - 1

        # Ensure indices are within bounds
        if plot_index >= plot_effect_samples.shape[2]:
            logger.warning(f"Skipping index out of bounds: entry_index={entry_index}, plot_index={plot_index}")
            continue

        # Calculate probabilities for each sample
        probs_samples = []
        for chain in range(plot_effect_samples.shape[0]):
            for sample in range(plot_effect_samples.shape[1]):
                theta = (plot_effect_samples[chain, sample, plot_index] + 
                         temp_effect_samples[chain, sample, entry_index, plot_index])
                beta = beta_samples[chain, sample, rater_index]
                tau = tau_samples[chain, sample]
                probs = calculate_probs(theta, beta, tau)
                probs_samples.append(probs)

        all_probs.append(np.mean(probs_samples, axis=0))

    all_probs = np.array(all_probs)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the PyMC model fitting process.")
    parser.add_argument("config_file", type=str, help=f"Configuration file name located in {CONFIG_DIR}")
    args = parser.parse_args()
    main(args.config_file)
