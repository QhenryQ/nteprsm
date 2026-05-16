// Description: Ablation model derived from spatial_model.stan.
// It keeps the original spatial-only structure and plot effect GP, but
// replaces the shared-threshold plus rating-event-adjustment structure with
// rater-specific thresholds applied consistently across rating events.

functions {
  // Converts ratings onto a probability scale using rater-specific thresholds.
  // theta: latent turfgrass quality at the time of rating
  // tau: rater-specific thresholds
  real rsm(int y, real theta, vector tau) {
    vector[rows(tau) + 1] unsummed;
    vector[rows(tau) + 1] probs;
    unsummed = append_row(rep_vector(0, 1), theta - tau);
    probs = softmax(cumulative_sum(unsummed));
    return categorical_lpmf(y + 1 | probs);
  }
}

data {
  int<lower=1> N;                           // total number of responses
  int<lower=1> num_entries;                 // total number of entries
  int<lower=1> num_plots;                   // total number of plots
  int<lower=1> num_raters;                  // total number of raters
  int<lower=1> y_max;                       // max value of y when indexed from 0
  array[N] int<lower=1, upper=num_entries> entry_code; // entry code of response n
  array[N] int<lower=1, upper=num_plots> plot_code;    // plot code of response n
  array[N] int<lower=1, upper=num_raters> rater_code;  // rater code of response n
  array[N] int<lower=0, upper=y_max> y;                // rating quality
  matrix[num_plots, num_plots] DIST;                   // distance matrix for plots

  // held-out test data
  int<lower=1> N_test;
  array[N_test] int<lower=0, upper=y_max> y_test;
  array[N_test] int<lower=1, upper=num_entries> entry_code_test;
  array[N_test] int<lower=1, upper=num_plots> plot_code_test;
  array[N_test] int<lower=1, upper=num_raters> rater_code_test;
}

parameters {
  array[num_raters] vector[y_max] tau_rater;
  vector[num_entries] entry_effect;
  vector[num_plots] eta;              // normal(0, 1) trick
  real<lower=0> sigma_entry;          // entry effect variance
  real<lower=0> sigma_e;
  real<lower=0> inv_rho;
  real<lower=0> alpha;
}

transformed parameters {
  vector[N] theta;                // latent turfgrass quality
  vector[num_plots] plot_effect;  // plot effect

  {
    matrix[num_plots, num_plots] KERNEL;
    real sq_sigma_e = square(sigma_e);

    for (i in 1:(num_plots - 1)) {
      for (j in (i + 1):num_plots) {
        KERNEL[i, j] = square(alpha) * exp(-0.5 * square(DIST[i, j] * inv_rho));
        KERNEL[j, i] = KERNEL[i, j];
      }
    }
    for (i in 1:num_plots)
      KERNEL[i, i] = square(alpha) + sq_sigma_e;

    matrix[num_plots, num_plots] L_KERN = cholesky_decompose(KERNEL);
    plot_effect = L_KERN * eta;
  }

  for (n in 1:N) {
    theta[n] = entry_effect[entry_code[n]] + plot_effect[plot_code[n]];
  }
}

model {
  for (i in 1:num_raters)
    target += normal_lpdf(tau_rater[i] | 0, 5);

  entry_effect ~ normal(0, sigma_entry);
  sigma_entry ~ student_t(3, 0, 1);
  sigma_e ~ student_t(3, 0, 1);
  eta ~ normal(0, 1);
  alpha ~ student_t(3, 0, 1);
  inv_rho ~ gamma(5, 5);

  for (n in 1:N) {
    target += rsm(y[n], theta[n], tau_rater[rater_code[n]]);
  }
}

generated quantities {
  vector[N] log_lik;
  vector[N_test] log_lik_test;
  array[N_test] int y_rep_test;
  for (n in 1:N) {
    log_lik[n] = rsm(y[n], theta[n], tau_rater[rater_code[n]]);
  }
  for (n in 1:N_test) {
    vector[y_max + 1] probs_test = softmax(
      cumulative_sum(
        append_row(
          rep_vector(0, 1),
          entry_effect[entry_code_test[n]] + plot_effect[plot_code_test[n]]
          - tau_rater[rater_code_test[n]]
        )
      )
    );
    log_lik_test[n] = categorical_lpmf(y_test[n] + 1 | probs_test);
    y_rep_test[n] = categorical_rng(probs_test) - 1;
  }
}