import pymc as pm
import numpy as np
import arviz as az
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
    additional_data = config.get("additional_data", {})
    datahandler.generate_stan_data(**additional_data)
    return datahandler.stan_data, config

def create_pymc_model(data):
    """
    Create a PyMC model based on the preprocessed data.
    """
    with pm.Model() as model:
        # Define priors
        beta_free = pm.Normal("beta_free", mu=0, sigma=2, shape=data['num_raters']-1)
        tau_free = pm.Normal("tau_free", mu=0, sigma=2, shape=data['num_categories']-1)
        sigma = pm.HalfStudentT("sigma", nu=3, sigma=1)
        entry = pm.Normal("entry", mu=0, sigma=sigma, shape=data['num_entries'])
        eta = pm.Normal("eta", mu=0, sigma=1, shape=data['num_plots'])
        sigma_e = pm.HalfStudentT("sigma_e", nu=3, sigma=1)
        alpha = pm.HalfStudentT("alpha", nu=3, sigma=1)
        inv_rho = pm.Gamma("inv_rho", alpha=5, beta=5)

        # Transform parameters
        beta = pm.Deterministic("beta", pt.concatenate([beta_free, -pt.sum(beta_free, keepdims=True)]))
        tau = pm.Deterministic("tau", pt.concatenate([tau_free, -pt.sum(tau_free, keepdims=True)]))

        # Define Gaussian Process for plot effects using the distance matrix
        DIST = pt.as_tensor_variable(data['DIST'])
        cov = alpha**2 * pt.exp(-0.5 * (DIST * inv_rho)**2) + pt.eye(data['num_plots']) * sigma_e**2
        plot_effect = pm.MvNormal("plot_effect", mu=0, cov=cov, shape=data['num_plots'])

        # Adjusted turf quality
        theta = entry[data['entry_id']-1] + plot_effect[data['plot_id']-1]

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

def calculate_probs(theta, beta, tau):
    # Debugging information
    print(f"theta shape before reshape: {theta.shape}")
    print(f"beta shape before reshape: {beta.shape}")
    print(f"tau shape before reshape: {tau.shape}")
    
    # Reshape theta to ensure correct broadcasting
    theta = theta[:, np.newaxis]  # Shape: (1500, 1)
    beta = beta[:, np.newaxis] if beta.ndim == 1 else beta[:, np.newaxis, :]  # Ensure correct shape
    tau = tau[:, np.newaxis, :]  # Shape: (1500, 1, 9)
    
    # Debugging information after reshaping
    print(f"theta shape after reshape: {theta.shape}")
    print(f"beta shape after reshape: {beta.shape}")
    print(f"tau shape after reshape: {tau.shape}")

    unsummed = np.concatenate([np.zeros((theta.shape[0], 1)), theta - beta - tau], axis=1)
    cumsum = np.cumsum(unsummed, axis=1)
    exp_cumsum = np.exp(cumsum - np.max(cumsum, axis=1, keepdims=True))
    return exp_cumsum / np.sum(exp_cumsum, axis=1, keepdims=True)

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

    # Extract samples for entry, plot effects, beta, and tau
    entry_samples = trace.posterior['entry'].values
    plot_effect_samples = trace.posterior['plot_effect'].values
    beta_samples = trace.posterior['beta'].values
    tau_samples = trace.posterior['tau'].values

    # Debugging information for shapes
    logger.debug(f"entry_samples shape: {entry_samples.shape}")
    logger.debug(f"plot_effect_samples shape: {plot_effect_samples.shape}")
    logger.debug(f"data['entry_id'] shape: {data['entry_id'].shape}")
    logger.debug(f"data['plot_id'] shape: {data['plot_id'].shape}")
    logger.info(f"Unique entries: {len(np.unique(data['entry_id']))}")
    logger.info(f"Unique plots: {len(np.unique(data['plot_id']))}")

    # Function to calculate probabilities
    all_probs = []
    for n in range(len(data['entry_id'])):
        entry_index = data['entry_id'][n] - 1
        plot_index = data['plot_id'][n] - 1
        rater_index = data['rater_id'][n] - 1

        # Ensure indices are within bounds
        if entry_index >= entry_samples.shape[2] or plot_index >= plot_effect_samples.shape[2]:
            logger.warning(f"Skipping index out of bounds: entry_index={entry_index}, plot_index={plot_index}")
            continue

        # Extract the correct samples for entry and plot effects
        entry_sample = entry_samples[:, :, entry_index]  # Shape: (4, 1500)
        plot_effect_sample = plot_effect_samples[:, :, plot_index]  # Shape: (4, 1500)

        # Ensure both have the same shape before adding
        theta_samples = entry_sample + plot_effect_sample

        # Extract beta samples for the rater
        beta_samples_n = beta_samples[:, :, rater_index]

        # Calculate probabilities for each sample
        probs_samples = np.array([calculate_probs(theta, beta, tau) 
                                  for theta, beta, tau in zip(theta_samples, beta_samples_n, tau_samples)])
        
        all_probs.append(probs_samples.mean(axis=0))

    all_probs = np.array(all_probs)
    
    # Log average probability distributions for all observations
    logger.info("Average probability distributions for all observations:")
    logger.info(all_probs.mean(axis=0))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the PyMC model fitting process.")
    parser.add_argument("config_file", type=str, help=f"Configuration file name located in {CONFIG_DIR}")
    args = parser.parse_args()
    main(args.config_file)

