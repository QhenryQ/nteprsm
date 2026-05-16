// Description: Stan model for spatial Rasch model, in which we assume that the 
// raters are not consistent in the different rating events. We remove the 
// spatial effect from the rating event and model the spatial effect using a
// Gaussian process. 
functions {
  // Convert ratings onto a probability scale
  // theta: turf quality at the time of rating on latent scale
  // beta: rater severity
  // tau: rating thresholds
  real rsm(int y, real theta, real beta, vector tau) {
    vector[rows(tau) + 1] unsummed;
    vector[rows(tau) + 1] probs;
    unsummed = append_row(rep_vector(0, 1), theta - beta - tau);
    probs = softmax(cumulative_sum(unsummed));
    return categorical_lpmf(y + 1 | probs);
  }
}

data {
  int<lower=1> N;                           // number of responses
  int<lower=1> num_ratings;                 // number of rating events
  int<lower=1> num_entries;                 // number of entries
  int<lower=1> num_plots;                   // number of plots
  int<lower=1> y_max;                       // number of thresholds (y is 0-indexed, y_max = num_categories - 1)
  array[N] int<lower=1, upper=num_ratings> rating_event_code; // rating event code for y[n]
  array[N] int<lower=1, upper=num_entries> entry_code;        // entry code of y[n]
  array[N] int<lower=1, upper=num_plots> plot_code;           // plot id for y[n]
  array[N] int<lower=0> y;                                    // response is 0-indexed
  matrix[num_plots, num_plots] DIST;                          // distance matrix for all entries 
}

parameters {
  vector[num_ratings - 1] beta_free;
  vector[y_max - 1] tau_free;
  vector[num_entries] entry_effect;
  vector[num_plots] eta;              // normal(0, 1) trick
  real<lower=0> sigma_entry;          // entry effect variance
  real<lower=0> sigma_e;
  real<lower=0> inv_rho;
  real<lower=0> alpha;
}

transformed parameters {
  vector[N] theta;                   // adjusted turf quality 
  vector[num_ratings] beta;          // rating severity 
  vector[y_max] tau;                 // Rasch-Andrich threshold
  vector[num_plots] plot_effect;     // plot effect

  // Create kernel for GP() based on plot distance
  {
    matrix[num_plots, num_plots] KERNEL;
    real sq_sigma_e = square(sigma_e);

    // Create kernel for GP() using distance matrix 
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
  
  beta[1:(num_ratings - 1)] = beta_free;
  beta[num_ratings] = -1 * sum(beta_free);
  tau[1:(y_max - 1)] = tau_free;
  tau[y_max] = -1 * sum(tau_free);

  for (n in 1:N) {
    theta[n] = entry_effect[entry_code[n]] + plot_effect[plot_code[n]]; 
    // turf quality at rating is partitioned into entry inherit part 
    // + plot location effect where plot location effect is modeled using a GP()
  }
}

model {
  target += normal_lpdf(beta | 0, 2);
  target += normal_lpdf(tau | 0, 2);
  entry_effect ~ normal(0, sigma_entry);
  sigma_entry ~ student_t(3, 0, 1);
  sigma_e ~ student_t(3, 0, 1);
  eta ~ normal(0, 1);
  alpha ~ student_t(3, 0, 1);
  inv_rho ~ gamma(5, 5); 
  
  for (n in 1:N) {
    target += rsm(y[n], theta[n], beta[rating_event_code[n]], tau);
  }
}

// adding generated quantities block to compute log likelihood
generated quantities {
  vector[N] log_lik;
  for (n in 1:N) {
    log_lik[n] = rsm(y[n], theta[n], beta[rating_event_code[n]], tau);
  }
}
