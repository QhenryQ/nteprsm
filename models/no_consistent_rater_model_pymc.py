import pymc as pm
import numpy as np
import pandas as pd
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
        beta_free = pm.Normal("beta_free", mu=0, sigma=2, shape=data['I']-1)
        tau_free = pm.Normal("tau_free", mu=0, sigma=2, shape=data['M']-1)
        sigma = pm.HalfStudentT("sigma", nu=3, sigma=1)
        entry = pm.Normal("entry", mu=0, sigma=sigma, shape=data['J'])
        eta = pm.Normal("eta", mu=0, sigma=1, shape=data['P'])
        sigma_e = pm.HalfStudentT("sigma_e", nu=3, sigma=1)
        alpha = pm.HalfStudentT("alpha", nu=3, sigma=1)
        inv_rho = pm.Gamma("inv_rho", alpha=5, beta=5)

        # Transformed parameters
        beta = pm.Deterministic("beta", pt.concatenate([beta_free, -pt.sum(beta_free, keepdims=True)]))
        tau = pm.Deterministic("tau", pt.concatenate([tau_free, -pt.sum(tau_free, keepdims=True)]))

        # Gaussian Process for plot effects
        DIST = pt.as_tensor_variable(data['DIST'])
        cov = alpha**2 * pt.exp(-0.5 * (DIST * inv_rho)**2) + pt.eye(data['P']) * sigma_e**2
        plot_effect = pm.MvNormal("plot_effect", mu=0, cov=cov, shape=data['P'])

        # Theta (adjusted turf quality)
        theta = entry[data['jj']-1] + plot_effect[data['pp']-1]

        # Likelihood
        def rsm(theta, beta, tau):
            unsummed = pt.concatenate([pt.zeros((theta.shape[0], 1)), pt.outer(theta - beta, pt.ones_like(tau)) - tau], axis=1)
            cumsum = pt.cumsum(unsummed, axis=1)
            # Manual implementation of softmax
            exp_cumsum = pt.exp(cumsum - pt.max(cumsum, axis=1, keepdims=True))
            probs = exp_cumsum / pt.sum(exp_cumsum, axis=1, keepdims=True)
            return probs

        probs = rsm(theta, beta[data['ii']-1], tau)
        
        # Print shapes for debugging
        print(f"probs shape: {probs.eval().shape}")
        print(f"data['y'] shape: {np.array(data['y']).shape}")
        print(f"Number of categories (M): {data['M']}")

        y = pm.Categorical("y", p=probs, observed=data['y'])

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
    
    # Print data shapes for debugging
    print(f"Number of observations (N): {data['N']}")
    print(f"Number of ratings (I): {data['I']}")
    print(f"Number of entries (J): {data['J']}")
    print(f"Number of plots (P): {data['P']}")
    print(f"Number of categories (M): {data['M']}")
    print(f"Shape of ii: {np.array(data['ii']).shape}")
    print(f"Shape of jj: {np.array(data['jj']).shape}")
    print(f"Shape of pp: {np.array(data['pp']).shape}")
    print(f"Shape of y: {np.array(data['y']).shape}")
    print(f"Shape of DIST: {np.array(data['DIST']).shape}")
    
    model = create_pymc_model(data)
    
    print("Model structure:")
    print(model)
    
    trace = run_model(model, config)
    
    # Basic diagnostics
    summary = az.summary(trace)
    print(summary)
    
    # Extract posterior samples
    posterior_samples = az.extract(trace)
    
    # Save results to 4 CSV files, one for each chain
    output_dir = Path(config['sampling']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    date_str = pd.Timestamp.now().strftime("%Y%m%d")
    base_filename = f"no_consistent_rater_model_dist_matrix-{date_str}"
    
    for chain in range(4):  # Assuming 4 chains
        chain_data = {}
        for var_name, var_data in posterior_samples.items():
            if var_name in ['beta_free', 'tau_free', 'entry', 'plot_effect', 'beta', 'tau', 'sigma', 'sigma_e', 'alpha', 'inv_rho']:
                # Flatten the array and convert to 1D
                flattened_data = var_data.sel(chain=chain).values.flatten()
                # Create column names with indices
                for i, value in enumerate(flattened_data):
                    chain_data[f"{var_name}[{i}]"] = [value]
        
        # Calculate theta
        entry_samples = posterior_samples.entry.sel(chain=chain).values
        plot_effect_samples = posterior_samples.plot_effect.sel(chain=chain).values
        theta_samples = entry_samples[data['jj']-1] + plot_effect_samples[data['pp']-1]
        for i, value in enumerate(theta_samples.flatten()):
            chain_data[f"theta[{i}]"] = [value]
        
        # Add additional columns
        chain_data['lp__'] = trace.sample_stats.sel(chain=chain).lp.values
        chain_data['accept_stat__'] = trace.sample_stats.sel(chain=chain).accept.values
        chain_data['stepsize__'] = trace.sample_stats.sel(chain=chain).step_size.values
        chain_data['treedepth__'] = trace.sample_stats.sel(chain=chain).tree_depth.values
        chain_data['n_leapfrog__'] = trace.sample_stats.sel(chain=chain).n_steps.values
        chain_data['divergent__'] = trace.sample_stats.sel(chain=chain).diverging.values
        chain_data['energy__'] = trace.sample_stats.sel(chain=chain).energy.values
        
        df = pd.DataFrame(chain_data)
        
        filename = f"{base_filename}_{chain+1}.csv"
        df.to_csv(output_dir / filename, index=False)
        print(f"Chain {chain+1} results saved to {output_dir / filename}")

    
    # Save full trace for further analysis if needed
    az.to_netcdf(trace, filename=output_dir / "trace.netcdf")
    print(f"Full trace saved to {output_dir / 'trace.netcdf'}")

    # Example: calculate probabilities for the first observation
    n = 0  # first observation
    entry_samples = posterior_samples.entry.values
    plot_samples = posterior_samples.plot.values
    theta_samples = entry_samples[:, data['jj'][n]-1] + plot_samples[:, data['pp'][n]-1]
    beta_samples = posterior_samples.beta[:, data['ii'][n]-1].values
    tau_samples = posterior_samples.tau.values
    
    def calculate_probs(theta, beta, tau):
        unsummed = np.concatenate([np.zeros((1,)), theta - beta - tau])
        cumsum = np.cumsum(unsummed)
        exp_cumsum = np.exp(cumsum - np.max(cumsum))
        return exp_cumsum / np.sum(exp_cumsum)
    
    probs_samples = np.array([calculate_probs(theta, beta, tau) for theta, beta, tau in zip(theta_samples, beta_samples, tau_samples)])
    
    print("Probability distribution for first observation:")
    print(probs_samples.mean(axis=0))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the PyMC model fitting process.")
    parser.add_argument("config_file", type=str, help=f"Configuration file name located in {CONFIG_DIR}")
    args = parser.parse_args()
    main(args.config_file)