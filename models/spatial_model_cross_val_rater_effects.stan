// Description: Cross-validation variant of spatial_model.stan.
// This keeps the original shared-threshold rating structure, but replaces
// rating-event-specific shifts with persistent rater-specific shifts so
// held-out dates can reuse learned annotation behavior instead of sampling
// fresh beta_test values from the prior.
//
// Compatibility note:
// The Python data pipeline already supplies rating-event fields for the older
// cross-validation design. They are retained in the data block below so this
// file can be used with the existing runner without changing the data builder.

functions {
  // Converts ratings onto a probability scale.
  // theta: latent turfgrass quality at the time of rating
  // beta_rater: persistent rater-specific severity adjustment
  // tau: shared thresholds
  real rsm(int y, real theta, real beta_rater, vector tau) {
    vector[rows(tau) + 1] unsummed;
    vector[rows(tau) + 1] probs;
    unsummed = append_row(rep_vector(0, 1), theta - beta_rater - tau);
    probs = softmax(cumulative_sum(unsummed));
    return categorical_lpmf(y + 1 | probs);
  }
}

data {
  int<lower=1> N;                           // total number of responses
  int<lower=1> num_ratings;                 // compatibility input, unused here
  int<lower=1> num_entries;                 // total number of entries
  int<lower=1> num_plots;                   // total number of plots
  int<lower=1> num_raters;                  // total number of raters
  int<lower=1> y_max;                       // max value of y when indexed from 0
  array[N] int<lower=1, upper=num_ratings> rating_event_code; // compatibility input, unused here
  array[N] int<lower=1, upper=num_entries> entry_code;        // entry code of response n
  array[N] int<lower=1, upper=num_plots> plot_code;           // plot code of response n
  array[N] int<lower=1, upper=num_raters> rater_code;         // rater code of response n
  array[N] int<lower=0> y;                                    // rating quality
  matrix[num_plots, num_plots] DIST;                          // distance matrix for plots

  // held-out test data
  int<lower=1> N_test;
  int<lower=1> num_ratings_test; // compatibility input, unused here
  array[N_test] int<lower=0, upper=y_max> y_test;
  array[N_test] int<lower=1, upper=num_ratings_test> rating_event_code_test; // compatibility input, unused here
  array[N_test] int<lower=1, upper=num_entries> entry_code_test;
  array[N_test] int<lower=1, upper=num_plots> plot_code_test;
  array[N_test] int<lower=1, upper=num_raters> rater_code_test;
}

parameters {
  vector[num_raters - 1] beta_rater_free;
  vector[y_max - 1] tau_free;
  vector[num_entries] entry_effect;
  vector[num_plots] eta;              // normal(0, 1) trick
  real<lower=0> sigma_entry;          // entry effect variance
  real<lower=0> sigma_e;
  real<lower=0> inv_rho;
  real<lower=0> alpha;
}

transformed parameters {
  vector[N] theta;                    // latent turfgrass quality
  vector[num_raters] beta_rater;      // persistent rater adjustment
  vector[y_max] tau;                  // shared thresholds
  vector[num_plots] plot_effect;      // plot effect

  // Create the kernel for the plot effect GP using plot distances.
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

  beta_rater[1:(num_raters - 1)] = beta_rater_free;
  beta_rater[num_raters] = -1 * sum(beta_rater_free);
  tau[1:(y_max - 1)] = tau_free;
  tau[y_max] = -1 * sum(tau_free);

  for (n in 1:N) {
    theta[n] = entry_effect[entry_code[n]] + plot_effect[plot_code[n]];
  }
}

model {
  // Relative to spatial_model.stan, the key change is that beta is now tied
  // to raters rather than rating events. This makes held-out dates re-use
  // learned rater behavior instead of introducing fresh prior noise.
  target += normal_lpdf(beta_rater | 0, 2);
  target += normal_lpdf(tau | 0, 2);
  entry_effect ~ normal(0, sigma_entry);
  sigma_entry ~ student_t(3, 0, 1);
  sigma_e ~ student_t(3, 0, 1);
  eta ~ normal(0, 1);
  alpha ~ student_t(3, 0, 1);
  inv_rho ~ gamma(5, 5);

  for (n in 1:N) {
    target += rsm(y[n], theta[n], beta_rater[rater_code[n]], tau);
  }
}

generated quantities {
  vector[N] log_lik;
  vector[N_test] log_lik_test;
  array[N_test] int y_rep_test;

  for (n in 1:N) {
    log_lik[n] = rsm(y[n], theta[n], beta_rater[rater_code[n]], tau);
  }
  for (n in 1:N_test) {
    vector[y_max + 1] probs_test = softmax(
      cumulative_sum(
        append_row(
          rep_vector(0, 1),
          entry_effect[entry_code_test[n]] + plot_effect[plot_code_test[n]]
          - beta_rater[rater_code_test[n]] - tau
        )
      )
    );
    log_lik_test[n] = categorical_lpmf(y_test[n] + 1 | probs_test);
    y_rep_test[n] = categorical_rng(probs_test) - 1;
  }
}