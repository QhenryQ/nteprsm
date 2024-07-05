import pymc as pm
import numpy as np
import pandas as pd
import pytensor.tensor as pt
import arviz as az
from pathlib import Path
from nteprsm import utils
from settings import CONFIG_DIR, LOG_DIR

logger = utils.setup_logging(LOG_DIR)

def load_and_preprocess_data(config_file):
    config = utils.load_config(config_file)
    datahandler = utils.DataHandler(filepath=config["data_path"], logger=logger)
    datahandler.load_data()
    datahandler.preprocess_data()
    datahandler.generate_stan_data(**config["stan_additional_data"])
    return datahandler.stan_data, config

def create_pymc_model(data):
    with pm.Model() as model:
        # Priors
        beta_free = pm.Normal("beta_free", mu=0, sigma=2, shape=data['num_raters']-1, 
                              initval=np.random.normal(0, 0.1, size=data['num_raters']-1))
        tau_free = pm.Normal("tau_free", mu=0, sigma=2, shape=data['num_categories']-1,
                             initval=np.random.normal(0, 0.1, size=data['num_categories']-1))
        sigma = pm.HalfStudentT("sigma", nu=3, sigma=1, 
                                initval=np.abs(np.random.normal(0, 0.1)))
        entry = pm.Normal("entry", mu=0, sigma=sigma, shape=data['num_entries'],
                          initval=np.random.normal(0, 0.1, size=data['num_entries']))
        eta = pm.Normal("eta", mu=0, sigma=1, shape=data['num_plots'],
                        initval=np.random.normal(0, 0.1, size=data['num_plots']))
        sigma_e = pm.HalfStudentT("sigma_e", nu=3, sigma=1,
                                  initval=np.abs(np.random.normal(0, 0.1)))
        alpha = pm.HalfStudentT("alpha", nu=3, sigma=1,
                                initval=np.abs(np.random.normal(0, 0.1)))
        inv_rho = pm.Gamma("inv_rho", alpha=5, beta=5,
                           initval=np.abs(np.random.normal(5, 1)))

        # Transformed parameters
        beta = pm.Deterministic("beta", pt.concatenate([beta_free, -pt.sum(beta_free, keepdims=True)]))
        tau = pm.Deterministic("tau", pt.concatenate([tau_free, -pt.sum(tau_free, keepdims=True)]))

        # Gaussian Process for plot effects
        DIST = pt.as_tensor_variable(data['DIST'])
        cov = alpha**2 * pt.exp(-0.5 * (DIST * inv_rho)**2) + pt.eye(data['num_plots']) * sigma_e**2
        plot = pm.MvNormal("plot", mu=0, cov=cov, shape=data['num_plots'])

        # Theta (adjusted turf quality)
        theta = entry[data['entry_id']-1] + plot[data['plot_id']-1]

        # Likelihood
        cutpoints = pt.sort(tau)  # Ensure cutpoints are ordered
        eta = theta - beta[data['rater_id']-1]
        y = pm.OrderedLogistic("y", cutpoints=cutpoints, eta=eta, observed=data['y'])

    return model

def run_model(model, config):
    with model:
        # Extract relevant parameters for PyMC
        nuts_kwargs = {
            'target_accept': config['sampling'].get('adapt_delta', 0.99),
            'max_treedepth': config['sampling'].get('max_treedepth', 15),
        }
        
        sample_kwargs = {
            'draws': config['sampling'].get('iter_sampling', 1500),
            'tune': config['sampling'].get('iter_warmup', 500),
            'chains': config['sampling'].get('parallel_chains', 4),
            'return_inferencedata': True,
            'random_seed': config['sampling'].get('seed', None),
            'progressbar': config['sampling'].get('show_progress', True),
            'init': 'adapt_diag',  # Use a more robust initialization
            'nuts': nuts_kwargs,  # Pass NUTS parameters as a nested dictionary
        }
        
        # Run the sampling
        try:
            trace = pm.sample(**sample_kwargs)
        except Exception as e:
            print(f"Sampling failed with error: {str(e)}")
            print("Attempting to provide more information about the model...")
            
            # Print information about the model
            print("Model variables:")
            for name, var in model.named_vars.items():
                print(f"{name}: {var}")
            
            # Try to evaluate the log probability
            try:
                print("Log probability at the starting point:")
                print(model.compile_logp()(model.initial_point()))
            except Exception as inner_e:
                print(f"Failed to evaluate log probability: {str(inner_e)}")
            
            # Check for infinite or NaN values in the data
            for name, var in model.named_vars.items():
                if hasattr(var, 'get_value'):
                    value = var.get_value()
                    if np.any(np.isnan(value)) or np.any(np.isinf(value)):
                        print(f"Warning: {name} contains NaN or infinite values")
            
            raise
    
    return trace
    
def main(config_file):
    data, config = load_and_preprocess_data(config_file)
    model = create_pymc_model(data)
    
    # Print model structure before sampling
    print("Model structure:")
    print(model)
    
    trace = run_model(model, config)
    
    # Basic diagnostics
    print(az.summary(trace))
    
    # Save results using ArviZ
    output_dir = Path(config['sampling']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    az.to_netcdf(trace, filename=output_dir / "trace.netcdf")

    print(f"Model results saved to {output_dir / 'trace.netcdf'}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the PyMC model fitting process.")
    parser.add_argument("config_file", type=str, help=f"Configuration file name located in {CONFIG_DIR}")
    args = parser.parse_args()
    main(args.config_file)