"""
PyMC translation of annual_seasonality_model.stan

Spatial effect: GP with periodic exponentiated-quadratic kernel, approximated
via 2-D real FFT (matches the gptools RFFT2 approach in the Stan model).

Temporal effect: Hilbert-space basis-function approximation of a periodic GP
(same cos/sin basis used in the Stan model).

Rater model: Rating Scale Model (RSM) with rater-specific thresholds and
discrimination fixed to 1.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import arviz as az
from pathlib import Path

from nteprsm import utils
from settings import CONFIG_DIR, LOG_DIR
from datetime import datetime

logger = utils.setup_logging(LOG_DIR)


# ---------------------------------------------------------------------------
# Helper functions (mirror the Stan functions block)
# ---------------------------------------------------------------------------

def rsm_logp(y, theta, tau):
    """
    Log-probability of an ordinal response under the Rating Scale Model.

    Parameters
    ----------
    y : int tensor, shape (N,)
        Observed ratings, 0-indexed (0 … y_max).
    theta : float tensor, shape (N,)
        Latent quality at time of rating.
    tau : float tensor, shape (N, y_max)
        Rater-specific thresholds for each observation.

    Returns
    -------
    logp : float tensor, shape (N,)
    """
    # unsummed[n, 0] = 0;  unsummed[n, k] = theta[n] - tau[n, k-1]  for k=1..y_max
    zeros = pt.zeros((theta.shape[0], 1))
    unsummed = pt.concatenate([zeros, theta[:, None] - tau], axis=1)  # (N, y_max+1)
    cum = pt.cumsum(unsummed, axis=1)  # (N, y_max+1)
    log_probs = cum - pt.logsumexp(cum, axis=1, keepdims=True)  # log-softmax
    return log_probs[pt.arange(y.shape[0]), y]


def diagSPD_periodic(alpha, rho, M):
    """
    Spectral densities for the periodic kernel Hilbert basis.

    Parameters
    ----------
    alpha : scalar tensor
        GP marginal standard deviation.
    rho : scalar tensor
        GP lengthscale.
    M : int
        Number of basis functions (before doubling for cos+sin).

    Returns
    -------
    spd : tensor, shape (2*M,)
    """
    a = 1.0 / rho ** 2
    m_vals = pt.arange(1, M + 1, dtype="float64")
    log_q = pt.log(alpha) + 0.5 * (pt.log(2.0) - a + pt.log(pt.iv(m_vals, a)))
    q = pt.exp(log_q)
    return pt.concatenate([q, q])


def PHI_periodic(x, M, w0):
    """
    Periodic Hilbert basis evaluated at `x`.

    Parameters
    ----------
    x : array-like, shape (T,)
        Input locations (standardised time).
    M : int
        Number of basis functions (before doubling).
    w0 : float
        Base frequency = 2*pi / period.

    Returns
    -------
    PHI : ndarray, shape (T, 2*M)
    """
    m_vals = np.arange(1, M + 1)
    # (T, M) outer product
    mw0x = np.outer(x, m_vals * w0)
    return np.concatenate([np.cos(mw0x), np.sin(mw0x)], axis=1)


def rfft2_periodic_exp_quad_cov(num_rows, num_cols, sigma, lengthscale):
    """
    Real-FFT2 covariance for a periodic exponentiated-quadratic kernel on a
    2-D grid, matching gptools' gp_periodic_exp_quad_cov_rfft2.

    Parameters
    ----------
    num_rows, num_cols : int
        Padded grid dimensions.
    sigma : scalar tensor
        GP marginal standard deviation.
    lengthscale : scalar tensor
        Isotropic lengthscale.

    Returns
    -------
    rfft2_cov : tensor, shape (num_rows, num_cols // 2 + 1)
        Power spectral density values on the RFFT2 frequency grid.
    """
    # Frequencies along each axis (matching numpy rfft2 convention)
    freq_rows = pt.arange(num_rows, dtype="float64")
    freq_cols = pt.arange(num_cols // 2 + 1, dtype="float64")

    # Map to angular distance in grid units, accounting for periodicity
    # dist_row[i] = min(i, num_rows - i)
    dist_rows = pt.minimum(freq_rows, num_rows - freq_rows)
    dist_cols = freq_cols  # rfft2 only stores 0 … N//2

    # Squared-distance grid
    dr2 = dist_rows[:, None] ** 2
    dc2 = dist_cols[None, :] ** 2
    sq_dist = dr2 + dc2  # (num_rows, num_cols//2 + 1)

    # Spectral density of exp-quad kernel: sigma^2 * exp(-0.5 * sq_dist / ls^2)
    # scaled by grid size so that iRFFT2(sqrt(psd) * z) has the right variance
    psd = sigma ** 2 * pt.exp(-0.5 * sq_dist / lengthscale ** 2) * (num_rows * num_cols)
    return psd


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(config_file: str):
    """Load and preprocess data, returning the stan_data dict."""
    config = utils.load_config(config_file)
    datahandler = utils.DataHandler(filepath=config["data_path"], logger=logger)
    datahandler.load_data()
    datahandler.preprocess_data()
    additional_data = config.get("stan_additional_data", {})
    datahandler.generate_stan_data(**additional_data)
    return datahandler.stan_data, config


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def create_model(data: dict):
    """
    Build the PyMC model corresponding to annual_seasonality_model.stan.

    Parameters
    ----------
    data : dict
        Stan-format data dictionary from DataHandler.generate_stan_data().
        Required keys: N, y, y_max, num_plots, num_rows, num_cols, plot_code,
        plot_row, plot_col, num_entries, num_ratings, M_f, entry_code,
        rating_event_code, time, num_raters, rater_code, pred_N, padding.

    Returns
    -------
    pymc.Model
    """
    # Unpack data -----------------------------------------------------------
    N = data["N"]
    y = data["y"]                       # 0-indexed, shape (N,)
    y_max = data["y_max"]               # number of thresholds = max category
    num_plots = data["num_plots"]
    num_rows = data["num_rows"]
    num_cols = data["num_cols"]
    padding = data.get("padding", 5)
    plot_code = data["plot_code"] - 1   # Stan is 1-indexed → Python 0-indexed
    plot_row = data["plot_row"] - 1
    plot_col = data["plot_col"] - 1
    num_entries = data["num_entries"]
    num_ratings = data["num_ratings"]
    M_f = data["M_f"]
    entry_code = data["entry_code"] - 1
    rating_event_code = data["rating_event_code"] - 1
    time_arr = np.array(data["time"], dtype=np.float64)  # length num_ratings
    num_raters = data["num_raters"]
    rater_code = data["rater_code"] - 1
    pred_N = data.get("pred_N", 100)

    # Transformed data (mirrors Stan transformed data block) ----------------
    num_rows_padded = num_rows + padding
    num_cols_padded = num_cols + padding

    mean_time = float(np.mean(time_arr))
    sd_time = float(np.std(time_arr, ddof=0))
    period = 1.0 / sd_time
    w0 = 2.0 * np.pi / period

    xn = (time_arr - mean_time) / sd_time       # (num_ratings,)
    PHI_f = PHI_periodic(xn, M_f, w0)           # (num_ratings, 2*M_f)

    pred_time = np.linspace(1.0 / pred_N, 1.0, pred_N)
    pred_xn = (pred_time - mean_time) / sd_time
    pred_PHI_f = PHI_periodic(pred_xn, M_f, w0)  # (pred_N, 2*M_f)

    # Build model -----------------------------------------------------------
    with pm.Model() as model:

        # --- Plot Effect GP (FFT2 approximation) --------------------------
        sigma_plot = pm.HalfNormal("sigma_plot", sigma=3.0)
        lengthscale_plot = pm.InverseGamma("lengthscale_plot", alpha=5.0, beta=3.0)

        # Whitened latent field on padded grid
        z = pm.Normal("z", mu=0.0, sigma=1.0,
                       shape=(num_rows_padded, num_cols_padded))

        # Power spectral density on RFFT2 grid
        psd = rfft2_periodic_exp_quad_cov(
            num_rows_padded, num_cols_padded, sigma_plot, lengthscale_plot
        )

        # Spectral-domain multiplication then inverse FFT
        z_rfft2 = pt.fft.rfft2(z)                         # complex
        sqrt_psd = pt.sqrt(psd)
        # rfft2 returns shape (rows, cols//2+1) — multiply element-wise
        scaled = z_rfft2 * sqrt_psd
        f_grid = pt.fft.irfft2(scaled, s=(num_rows_padded, num_cols_padded))

        # Extract plot effects at actual plot positions
        plot_effect = f_grid[plot_row, plot_col]  # (num_plots,)

        # --- Time Effect GP (Hilbert basis) --------------------------------
        sigma_f = pm.HalfNormal("sigma_f", sigma=3.0)
        lengthscale_f = pm.InverseGamma("lengthscale_f", alpha=5.0, beta=3.0)

        intercept_f = pm.Normal("intercept_f", mu=0.0, sigma=2.0,
                                shape=(num_entries,))
        beta_f = pm.Normal("beta_f", mu=0.0, sigma=2.0,
                           shape=(num_entries, 2 * M_f))

        # Spectral density weights
        diagSPD_f = diagSPD_periodic(sigma_f, lengthscale_f, M_f)  # (2*M_f,)

        # Weighted basis coefficients: (num_entries, 2*M_f)
        weighted_beta = beta_f * diagSPD_f[None, :]

        # time_effect[i, t] = intercept_f[i] + PHI_f[t, :] @ weighted_beta[i, :].T
        # Shape: (num_entries, num_ratings)
        time_effect = intercept_f[:, None] + pt.dot(weighted_beta, PHI_f.T)

        # --- Rater thresholds ----------------------------------------------
        tau_rater = pm.Normal("tau_rater", mu=0.0, sigma=5.0,
                              shape=(num_raters, y_max))

        # --- Likelihood ----------------------------------------------------
        # theta[n] = plot_effect[plot[n]] + time_effect[entry[n], event[n]]
        theta = (plot_effect[plot_code]
                 + time_effect[entry_code, rating_event_code])  # (N,)

        # Gather rater thresholds per observation: (N, y_max)
        tau_obs = tau_rater[rater_code]

        # Custom log-likelihood
        logp = rsm_logp(pt.as_tensor_variable(y), theta, tau_obs)
        pm.Potential("rsm_likelihood", logp.sum())

        # --- Generated quantities (deterministic) -------------------------
        pred_time_effect = pm.Deterministic(
            "pred_time_effect",
            intercept_f[:, None] + pt.dot(weighted_beta, pred_PHI_f.T)
        )  # (num_entries, pred_N)

        # Pointwise log-likelihood for LOO
        pm.Deterministic("log_lik", logp)  # (N,)

    return model


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def run_model(model, config):
    """Sample from the model using NUTS."""
    with model:
        nuts_kwargs = utils.get_nuts_kwargs(config)
        sample_kwargs = utils.get_sample_kwargs(config, nuts_kwargs)
        trace = pm.sample(**sample_kwargs)
    return trace


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config_file: str):
    """End-to-end: load data, build model, sample, save trace."""
    data, config = load_data(config_file)
    model = create_model(data)
    trace = run_model(model, config)

    # Save trace
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config.get("output_dir", "data/model_output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"trace_{timestamp}.netcdf"
    trace.to_netcdf(str(output_path))
    logger.info(f"Trace saved to {output_path}")

    return trace


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python pymc_annual_seasonality_model.py <config_file>")
        sys.exit(1)
    main(sys.argv[1])
