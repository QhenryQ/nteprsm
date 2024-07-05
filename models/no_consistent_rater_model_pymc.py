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
        beta_free = pm.Normal("beta_free", mu=0, sigma=2, shape=data['num_raters']-1)
        tau_free = pm.Normal("tau_free", mu=0, sigma=2, shape=data['num_categories']-1)
        sigma = pm.HalfStudentT("sigma", nu=3, sigma=1)
        entry = pm.Normal("entry", mu=0, sigma=sigma, shape=data['num_entries'])
        eta = pm.Normal("eta", mu=0, sigma=1, shape=data['num_plots'])
        sigma_e = pm.HalfStudentT("sigma_e", nu=3, sigma=1)
        alpha = pm.HalfStudentT("alpha", nu=3, sigma=1)
        inv_rho = pm.Gamma("inv_rho", alpha=5, beta=5)

        # Transformed parameters
        beta = pm.Deterministic("beta", pt.concatenate([beta_free, -pt.sum(beta_free, keepdims=True)]))
        tau = pm.Deterministic("tau", pt.concatenate([tau_free, -pt.sum(tau_free, keepdims=True)]))

        # Gaussian Process for plot effects
        DIST = pt.as_tensor_variable(data['DIST'])
        cov = alpha**2 * pt.exp(-0.5 * (DIST * inv_rho)**2) + pt.eye(data['num_plots']) * sigma_e**2
        plot_effect = pm.MvNormal("plot_effect", mu=0, cov=cov, shape=data['num_plots'])

        # Theta (adjusted turf quality)
        theta = entry[data['entry_id']-1] + plot_effect[data['plot_id']-1]

        # Likelihood
        def rsm(theta, beta, tau):
            unsummed = pt.concatenate([pt.zeros((theta.shape[0], 1)), pt.outer(theta - beta, pt.ones_like(tau)) - tau], axis=1)
            cumsum = pt.cumsum(unsummed, axis=1)
            # Manual implementation of softmax
            exp_cumsum = pt.exp(cumsum - pt.max(cumsum, axis=1, keepdims=True))
            probs = exp_cumsum / pt.sum(exp_cumsum, axis=1, keepdims=True)
            return probs

        probs = rsm(theta, beta[data['rater_id']-1], tau)
        
        # Print shapes for debugging
        print(f"probs shape: {probs.eval().shape}")
        print(f"data['y'] shape: {np.array(data['y']).shape}")
        print(f"Number of categories (num_categories): {data['num_categories']}")

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
    print(f"Number of raters (num_raters): {data['num_raters']}")
    print(f"Number of entries (num_entries): {data['num_entries']}")
    print(f"Number of plots (num_plots): {data['num_plots']}")
    print(f"Number of categories (num_categories): {data['num_categories']}")
    print(f"Shape of rater_id: {np.array(data['rater_id']).shape}")
    print(f"Shape of entry_id: {np.array(data['entry_id']).shape}")
    print(f"Shape of plot_id: {np.array(data['plot_id']).shape}")
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
        theta_samples = entry_samples[data['entry_id']-1] + plot_effect_samples[data['plot_id']-1]
        for i, value in enumerate(theta_samples.flatten()):
            chain_data[f"theta[{i}]"] = [value]
        
        # Add additional columns
        chain_data['lp__'] = trace.sample_stats.sel(chain=chain).lp.values
        chain_data['accept_stat__'] = trace.sample_stats.sel(chain=chain).get('accept', trace.sample_stats.sel(chain=chain).get('acceptance_rate', [None])).values
        chain_data['stepsize__'] = trace.sample_stats.sel(chain=chain).step_size.values
        chain_data['treedepth__'] = trace.sample_stats.sel(chain=chain).tree_depth.values
        chain_data['n_leapfrog__'] = trace.sample_stats.sel(chain=chain).n_steps.values
        chain_data['divergent__'] = trace.sample_stats.sel(chain=chain).diverging.values
        chain_data['energy__'] = trace.sample_stats.sel(chain=chain).energy.values
        
        # Print lengths of arrays for debugging
        print(f"Chain {chain} array lengths:")
        for key, value in chain_data.items():
            print(f"{key}: {len(value)}")
        
        # Find the most common length
        lengths = [len(v) for v in chain_data.values()]
        most_common_length = max(set(lengths), key=lengths.count)
        
        # Filter out arrays that don't have the most common length
        filtered_chain_data = {k: v for k, v in chain_data.items() if len(v) == most_common_length}
        
        # Create DataFrame with filtered data
        df = pd.DataFrame(filtered_chain_data)
        
        filename = f"{base_filename}_{chain+1}.csv"
        df.to_csv(output_dir / filename, index=False)
        print(f"Chain {chain+1} results saved to {output_dir / filename}")
        
        # Print information about discarded columns
        discarded_columns = set(chain_data.keys()) - set(filtered_chain_data.keys())
        if discarded_columns:
            print(f"Warning: The following columns were discarded due to length mismatch:")
            for col in discarded_columns:
                print(f"  {col}: length {len(chain_data[col])}")
    
    # Save full trace for further analysis if needed
    az.to_netcdf(trace, filename=output_dir / "trace.netcdf")
    print(f"Full trace saved to {output_dir / 'trace.netcdf'}")

    # Calculate probabilities for all observations
    entry_samples = posterior_samples.entry.values
    plot_effect_samples = posterior_samples.plot_effect.values
    beta_samples = posterior_samples.beta.values
    tau_samples = posterior_samples.tau.values

    def calculate_probs(theta, beta, tau):
        unsummed = np.concatenate([np.zeros((1,)), theta - beta - tau])
        cumsum = np.cumsum(unsummed)
        exp_cumsum = np.exp(cumsum - np.max(cumsum))
        return exp_cumsum / np.sum(exp_cumsum)

    all_probs = []
    for n in range(len(data['entry_id'])):  # Loop over all observations
        entry_index = data['entry_id'][n] - 1
        plot_index = data['plot_id'][n] - 1
        rater_index = data['rater_id'][n] - 1

        theta_samples = entry_samples[:, entry_index] + plot_effect_samples[:, plot_index]
        beta_samples_n = beta_samples[:, rater_index]

        probs_samples = np.array([calculate_probs(theta, beta, tau) 
                                  for theta, beta, tau in zip(theta_samples, beta_samples_n, tau_samples)])
        
        all_probs.append(probs_samples.mean(axis=0))

    all_probs = np.array(all_probs)
    
    print("Average probability distributions for all observations:")
    print(all_probs.mean(axis=0))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the PyMC model fitting process.")
    parser.add_argument("config_file", type=str, help=f"Configuration file name located in {CONFIG_DIR}")
    args = parser.parse_args()
    main(args.config_file)