// Description: Stan model for spatial and temporal effects, in which we assume
// that raters are consistent across different rating events. The latent
// turfgrass quality is modeled as a function of the plot effect and the annual
// seasonality effect, with rater-specific thresholds in the rating model. The
// plot effect is modeled using a Gaussian process with an RBF kernel, and the
// annual seasonality effect is modeled as a Gaussian process with a periodic
// kernel.

functions {
  #include gptools/util.stan
  #include gptools/fft.stan

  // Converts ratings onto a probability scale using the Rasch Rating Scale model
  // theta: turfgrass quality at the time of rating
  // tau: rater-specific thresholds
  real rsm(int y, real theta, vector tau) {
    vector[rows(tau) + 1] unsummed;
    vector[rows(tau) + 1] probs;
    unsummed = append_row(rep_vector(0, 1), theta - tau);
    probs = softmax(cumulative_sum(unsummed));
    return categorical_lpmf(y + 1 | probs);
  }

  // Hilbert basis methods helper functions

  // Returns the spectral density corresponding to a periodic kernel
  // alpha: The variance of the kernel used
  // rho: The lengthscale of the kernel
  // M: Number of Hilbert basis functions
  vector diagSPD_periodic(real alpha, real rho, int M) {
    real a = 1 / rho^2;
    vector[M] q = exp(
      log(alpha) + 0.5 * (log(2) - a + to_vector(log_modified_bessel_first_kind(linspaced_int_array(M, 1, M), a)))
    );
    return append_row(q, q);
  }

  // Returns the evaluations of the Eigenfunctions of the periodic Hilbert Basis
  // N: Dimension of the vector x 
  // M: number of basis functions
  // w0: Frequency
  // x: The values we want to evaluate the Hilbert Basis on
  matrix PHI_periodic(int N, int M, real w0, vector x) {
    matrix[N, M] mw0x = diag_post_multiply(rep_matrix(w0 * x, M), linspaced_vector(M, 1, M));
    return append_col(cos(mw0x), sin(mw0x));
  }
}

data {  
  // general dataset information
  int<lower=1> N;                       // total number of responses
  int<lower=1> y_max;                   // max value of y when indexed from 0
  array[N] int<lower=0, upper=y_max> y; // rating quality

  // data needed for the plot effect GP
  int<lower=1> num_plots;          // total number of plots
  int<lower=1> num_rows;           // number of rows of the turf plot grid
  int<lower=1> num_cols;           // number of cols of the turf plot grid
  int padding;

  array[N] int<lower=1, upper=num_plots> plot_code; // use plot_code because plot identifiers are reused across trials
  array[num_plots] int<lower=1> plot_row;           // row of the plot corresponding to plot_code
  array[num_plots] int<lower=1> plot_col;           // column of the plot corresponding to plot_code

  // data needed for the time effect GP
  int<lower=1> num_entries;             // total number of entries
  int<lower=1> M_f;                     // number of basis functions
  int<lower=1> num_ratings;
  array[N] int<lower=1, upper=num_entries> entry_code; // entry code of response n
  array[N] int<lower=1> rating_event_code; // rating event of response n
  array[num_ratings] real time;          // time of year corresponding to each rating event, scaled from 0 to 1

  // data needed for rater information
  int<lower=1> num_raters;                            // total number of distinct raters
  array[N] int<lower=1, upper=num_raters> rater_code; // rater code of response n

  // data for predictions/generated quantities
  int<lower=1> pred_N;                  // total number of prediction times for the time effect
}

transformed data {
  // Time effect
  real mean_time = mean(time);
  real sd_time = sd(time);
  // pred_xn is the standardized vector of prediction times for the time effect GP
  vector[pred_N] pred_xn;
  real period = 1 / sd_time;
  int<lower=1> num_rows_padded = num_rows + padding;
  int<lower=1> num_cols_padded = num_cols + padding;
  vector[pred_N] pred_time;
  for (i in 1:pred_N) pred_time[i] = i * 1.0 / pred_N;

  // xn is the standardized vector of rating-event times
  vector[num_ratings] xn; 
  for (n in 1:num_ratings) xn[n] = (time[n]-mean_time)/sd_time;
  pred_xn = (pred_time - mean_time) / sd_time;
  matrix[num_ratings ,2*M_f] PHI_f = PHI_periodic(num_ratings, M_f, 2*pi()/period, xn);
}

parameters {
  // Plot effect Gaussian process
  real<lower=0> sigma_plot;                       // scale of the exponentiated quadratic GP for plots
  matrix[num_rows_padded, num_cols_padded] z;     // standard normal latent field
  real<lower=0> lengthscale_plot;                 // lengthscale of the plot effect

  // Time effect Gaussian process
  array[num_entries] real intercept_f;            // intercept of the GP
  array[num_entries] vector[2 * M_f] beta_f;      // basis function coefficients
  real<lower=0> lengthscale_f;                    // shared lengthscale of the GP
  real<lower=0> sigma_f;                          // shared scale of the GP

  // Rater parameters
  array[num_raters] vector[y_max] tau_rater;      // since y is 0-indexed, y_max is the number of thresholds
}

transformed parameters {
  vector[num_plots] plot_effect;                                 // plot effect
  array[num_entries] vector[num_ratings] time_effect;      // time effect
  vector[2 * M_f] diagSPD_f;                                     // spectral densities of periodic kernel

  // Fourier method for the plot effect
  matrix[num_rows_padded, num_cols_padded %/% 2 + 1] rfft2_cov =
    gp_periodic_exp_quad_cov_rfft2(
      num_rows_padded, num_cols_padded,
      sigma_plot, [lengthscale_plot, lengthscale_plot]',
      [num_rows_padded, num_cols_padded]'
    );
  matrix[num_rows_padded, num_cols_padded] f = gp_inv_rfft2(
    z, rep_matrix(0, num_rows_padded, num_cols_padded), rfft2_cov
  );
  for (i in 1:num_plots) plot_effect[i] = f[plot_row[i], plot_col[i]];

  // Hilbert basis approximation for the time effect
  diagSPD_f = diagSPD_periodic(sigma_f, lengthscale_f, M_f);
  for (i in 1:num_entries)
    time_effect[i] = intercept_f[i] + PHI_f * (diagSPD_f .* beta_f[i]);
}

model {
  // priors for the plot effect
  sigma_plot ~ normal(0, 3);
  lengthscale_plot ~ inv_gamma(5, 3);
  to_vector(z) ~ std_normal();

  // priors for the time effect
  for (i in 1:num_entries) {
    intercept_f[i] ~ normal(0, 2);    // intercept of the Gaussian process
    beta_f[i] ~ normal(0, 2);         // Hilbert basis coefficients
  }
  lengthscale_f ~ inv_gamma(5, 3);    // Gaussian process lengthscale parameter
  sigma_f ~ normal(0, 3);             // Gaussian process scale parameter

  // priors for the rater-specific thresholds
  for (i in 1:num_raters)
    target += normal_lpdf(tau_rater[i] | 0, 5);

  // Model the response y[n]
  for (n in 1:N)
    target += rsm(
      y[n],
      plot_effect[plot_code[n]] + time_effect[entry_code[n]][rating_event_code[n]],
      tau_rater[rater_code[n]]
    );
}

generated quantities {
  array[num_entries] vector[pred_N] pred_time_effect;
  matrix[pred_N, 2 * M_f] pred_PHI_f;
  vector[N] log_lik;

  pred_PHI_f = PHI_periodic(pred_N, M_f, 2 * pi() / period, pred_xn);
  for (i in 1:num_entries)
    pred_time_effect[i] = intercept_f[i] + pred_PHI_f * (diagSPD_f .* beta_f[i]);
  for (n in 1:N)
    log_lik[n] = rsm(
      y[n],
      plot_effect[plot_code[n]] + time_effect[entry_code[n]][rating_event_code[n]],
      tau_rater[rater_code[n]]
    );
}
