// Description: Ablation model derived from annual_seasonality_model.stan.
// It keeps the annual seasonality effect and approximated plot effect, but
// restores the shared-threshold plus rating-event-adjustment structure from
// spatial_model.stan.

functions {
  #include gptools/util.stan
  #include gptools/fft.stan

  // Converts ratings onto a probability scale using the original spatial-model
  // threshold structure.
  // theta: latent turfgrass quality at the time of rating
  // beta: rating-event adjustment
  // tau: shared thresholds
  real rsm(int y, real theta, real beta, vector tau) {
    vector[rows(tau) + 1] unsummed;
    vector[rows(tau) + 1] probs;
    unsummed = append_row(rep_vector(0, 1), theta - beta - tau);
    probs = softmax(cumulative_sum(unsummed));
    return categorical_lpmf(y + 1 | probs);
  }

  // Returns the spectral density corresponding to a periodic kernel.
  vector diagSPD_periodic(real alpha, real rho, int M) {
    real a = 1 / rho^2;
    vector[M] q = exp(
      log(alpha) + 0.5 * (log(2) - a + to_vector(log_modified_bessel_first_kind(linspaced_int_array(M, 1, M), a)))
    );
    return append_row(q, q);
  }

  // Returns the evaluations of the Eigenfunctions of the periodic Hilbert basis.
  matrix PHI_periodic(int N, int M, real w0, vector x) {
    matrix[N, M] mw0x = diag_post_multiply(rep_matrix(w0 * x, M), linspaced_vector(M, 1, M));
    return append_col(cos(mw0x), sin(mw0x));
  }
}

data {
  int<lower=1> N;                       // total number of responses
  int<lower=1> y_max;                   // max value of y when indexed from 0
  array[N] int<lower=0, upper=y_max> y; // rating quality

  // data needed for the plot effect GP
  int<lower=1> num_plots;          // total number of plots
  int<lower=1> num_rows;           // number of rows of the turf plot grid
  int<lower=1> num_cols;           // number of cols of the turf plot grid
  int padding;

  array[N] int<lower=1, upper=num_plots> plot_code;
  array[num_plots] int<lower=1> plot_row;
  array[num_plots] int<lower=1> plot_col;

  // data needed for the time effect GP
  int<lower=1> num_entries;             // total number of entries
  int<lower=1> M_f;                     // number of basis functions
  int<lower=1> num_ratings;             // total number of rating events
  array[N] int<lower=1, upper=num_entries> entry_code;
  array[N] int<lower=1, upper=num_ratings> rating_event_code;
  array[num_ratings] real time;

  // data for predictions/generated quantities
  int<lower=1> pred_N;

  // held-out test data
  int<lower=1> N_test;
  int<lower=1> num_ratings_test;
  array[N_test] int<lower=0, upper=y_max> y_test;
  array[num_ratings_test] real time_test;
  array[N_test] int<lower=1, upper=num_entries> entry_code_test;
  array[N_test] int<lower=1, upper=num_plots> plot_code_test;
  array[N_test] int<lower=1, upper=num_ratings_test> rating_event_code_test;
}

transformed data {
  real mean_time = mean(time);
  real sd_time = sd(time);
  vector[pred_N] pred_xn;
  real period = 1 / sd_time;
  int<lower=1> num_rows_padded = num_rows + padding;
  int<lower=1> num_cols_padded = num_cols + padding;
  vector[pred_N] pred_time;
  for (i in 1:pred_N) pred_time[i] = i * 1.0 / pred_N;

  vector[num_ratings] xn;
  for (n in 1:num_ratings) xn[n] = (time[n] - mean_time) / sd_time;
  pred_xn = (pred_time - mean_time) / sd_time;
  matrix[num_ratings, 2 * M_f] PHI_f = PHI_periodic(num_ratings, M_f, 2 * pi() / period, xn);

  vector[num_ratings_test] xn_test;
  for (n in 1:num_ratings_test) xn_test[n] = (time_test[n] - mean_time) / sd_time;
  matrix[num_ratings_test, 2 * M_f] PHI_f_test = PHI_periodic(num_ratings_test, M_f, 2 * pi() / period, xn_test);
}

parameters {
  // Plot effect Gaussian process
  real<lower=0> sigma_plot;
  matrix[num_rows_padded, num_cols_padded] z;
  real<lower=0> lengthscale_plot;

  // Time effect Gaussian process
  array[num_entries] real intercept_f;
  array[num_entries] vector[2 * M_f] beta_f;
  real<lower=0> lengthscale_f;
  real<lower=0> sigma_f;

  // Shared-threshold and rating-event-adjustment parameters
  vector[num_ratings - 1] beta_free;
  vector[y_max - 1] tau_free;
}

transformed parameters {
  vector[num_plots] plot_effect;
  array[num_entries] vector[num_ratings] time_effect;
  vector[2 * M_f] diagSPD_f;
  vector[num_ratings] beta;
  vector[y_max] tau;

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

  beta[1:(num_ratings - 1)] = beta_free;
  beta[num_ratings] = -1 * sum(beta_free);
  tau[1:(y_max - 1)] = tau_free;
  tau[y_max] = -1 * sum(tau_free);
}

model {
  // priors for the plot effect
  sigma_plot ~ normal(0, 3);
  lengthscale_plot ~ inv_gamma(5, 3);
  to_vector(z) ~ std_normal();

  // priors for the time effect
  for (i in 1:num_entries) {
    intercept_f[i] ~ normal(0, 2);
    beta_f[i] ~ normal(0, 2);
  }
  lengthscale_f ~ inv_gamma(5, 3);
  sigma_f ~ normal(0, 3);

  // priors for the shared thresholds and rating-event adjustments
  target += normal_lpdf(beta | 0, 2);
  target += normal_lpdf(tau | 0, 2);

  // Model the response y[n]
  for (n in 1:N)
    target += rsm(
      y[n],
      plot_effect[plot_code[n]] + time_effect[entry_code[n]][rating_event_code[n]],
      beta[rating_event_code[n]],
      tau
    );
}

generated quantities {
  array[num_entries] vector[pred_N] pred_time_effect;
  matrix[pred_N, 2 * M_f] pred_PHI_f;
  vector[N] log_lik;
  array[num_entries] vector[num_ratings_test] time_effect_test;
  vector[N_test] log_lik_test;
  array[N_test] int y_rep_test;
  // Held-out rating events have no posterior for beta, so we propagate prior
  // uncertainty by drawing one beta_test per held-out rating event from the
  // training prior Normal(0, 2). This makes log_lik_test/y_rep_test honest
  // about the unknown event-level shift on unseen dates.
  vector[num_ratings_test] beta_test;
  for (k in 1:num_ratings_test) beta_test[k] = normal_rng(0, 2);

  pred_PHI_f = PHI_periodic(pred_N, M_f, 2 * pi() / period, pred_xn);
  for (i in 1:num_entries)
    pred_time_effect[i] = intercept_f[i] + pred_PHI_f * (diagSPD_f .* beta_f[i]);
  for (n in 1:N)
    log_lik[n] = rsm(
      y[n],
      plot_effect[plot_code[n]] + time_effect[entry_code[n]][rating_event_code[n]],
      beta[rating_event_code[n]],
      tau
    );

  for (i in 1:num_entries)
    time_effect_test[i] = intercept_f[i] + PHI_f_test * (diagSPD_f .* beta_f[i]);
  for (n in 1:N_test) {
    vector[y_max + 1] probs_test = softmax(
      cumulative_sum(
        append_row(
          rep_vector(0, 1),
          plot_effect[plot_code_test[n]] + time_effect_test[entry_code_test[n]][rating_event_code_test[n]]
          - beta_test[rating_event_code_test[n]] - tau
        )
      )
    );
    log_lik_test[n] = categorical_lpmf(y_test[n] + 1 | probs_test);
    y_rep_test[n] = categorical_rng(probs_test) - 1;
  }
}