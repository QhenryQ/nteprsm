functions {
  #include gptools/util.stan
  #include gptools/fft.stan
// Converts ratings onto a probability scale using the Rasch Rating Scale model
// theta: turfgrass quality at the time of rating
// tau: rating thresholds
  real rsm(int y, real theta, real beta, vector tau) {
    vector[rows(tau) + 1] unsummed;
    vector[rows(tau) + 1] probs;
    unsummed = append_row(rep_vector(0, 1), theta - beta - tau);
    probs = softmax(cumulative_sum(unsummed));
    return categorical_lpmf(y+1| probs);
  }
  
  // Hilbert basis methods helper functions
  
  // Returns the spectral density corresponding to a periodic kernel
  // alpha 	: The variance of the kernel used
  // rho 	: The lengthscale of the kernel
  // M 		: Number of Hilbert basis functions
  vector diagSPD_periodic(real alpha, real rho, int M) {
    real a = 1/rho^2;
    vector[M] q = exp(log(alpha) + 0.5 * (log(2) - a + to_vector(log_modified_bessel_first_kind(linspaced_int_array(M, 1, M), a))));
    return append_row(q,q);
  }
  
  // Returns the evaluations of the Eigenfunctions of the periodic Hilbert Basis
  // N : Dimension of the vector x 
  // M : number of basis functions
  // w0: Frequency
  // x : The values we want to take evaluate the Hilbert Basis on
  matrix PHI_periodic(int N, int M, real w0, vector x) {
    matrix[N,M] mw0x = diag_post_multiply(rep_matrix(w0*x, M), linspaced_vector(M, 1, M));
    return append_col(cos(mw0x), sin(mw0x));
  }
  
}

data {  
  int<lower=1> N;                			// total number of responses
  int<lower=1> num_raters;                  // total number of distinct raters
  int<lower=1> num_entries;                	// total number of entries
  int<lower=1> num_plots;                	// total number of plots
  int<lower=2> num_categories;              // number of categories
  array[N] int<lower=1,upper=num_raters> rater_id;    	// rater of response n
  array[N] int<lower=1,upper=num_entries> entry_id;    	// entry of response n
  array[N] int<lower=1,upper=num_plots> plot_id;    	// plot id for response n

  array[N] int<lower=0,upper=num_categories-1> y;         // Rating quality
  
// data needed for fourier method
  int<lower=1> num_ratings_per_entry;	// number of ratings each entry received
  int<lower=1> num_rows;         		// number of rows of the turf plot grid
  int<lower=1> num_cols;         		// number of cols of the turf plot grid
  int<lower=1> num_rows_padded;	 		// number of rows + padding
  int<lower=1> num_cols_padded;  		// number of columns + padding  
  array[num_plots] int<lower=1> plot_row;			 // row of the plot corresponding to plot_id
  array[num_plots] int<lower=1> plot_col;			 // column of the plot corresponding to plot_id
  
// data needed for time GP
  array[N] real time;					// time of year the rating was taken at (float from 0 to 1)
  int<lower=1> M_f;    					// number of basis functions
  array[N] int<lower=1> entry_cumcount; // cumulative count of the entry, e.g. if it is the 2nd time it appeared as an entry, it would be "2"
  
// data for predictions/generated quantitites
  int<lower=1> pred_N;        			// total number of generated time effect responses
  vector[pred_N] pred_time;				// time of year the generated rating was taken at (Multiply by 365 to get day of year)
}
transformed data {
  // time effect
  real mean_time = mean(time);
  real sd_time = sd(time);
  vector[pred_N] pred_xn;
  real period = 1/sd_time;
  
  // xn is standardized array of dates corresponding to each entry
  vector[num_ratings_per_entry] xn; 
  for (n in 1:N) xn[entry_cumcount[n]] = (time[n]-mean_time)/sd_time;
  pred_xn = (pred_time-mean_time)/sd_time;
}
parameters {
  // Area effect GPs
  real<lower=0> sigma_plot;			 			// variance of the exponentiated quadratic GP for plots
  matrix[num_rows_padded, num_cols_padded] z; 	// standard normal distribution
  real <lower=0> lengthscale_plot;		 		// lengthscale of plot
  
  // time GP
  array[num_entries] real intercept_f;			// intercept of the GP
  array[num_entries] vector[2*M_f] beta_f;      // basis functions coefficients
  real<lower=0> lengthscale_f; 					// shared lengthscale of GP
  real<lower=0> sigma_f;       					// shared scale(variance) of GP
  
  // rater parameters
  array[num_raters] vector[num_categories-2] tau_rater_free;			// Rasch-Andrich thresholds of each rater
  array[num_raters] real rater_severity;
}

transformed parameters {
  vector[num_plots] plot_effect;     							// plot effect
  array[num_entries] vector[num_ratings_per_entry] time_effect;	// time effect
  vector[2*M_f] diagSPD_f;										// spectral densities of periodic kernel
  matrix[num_ratings_per_entry ,2*M_f] PHI_f;					// evaluation of eigenfunctions of periodic kernel on x
  array[num_raters] vector[num_categories-1] tau_rater;
  
  // fourier method for plot GP
  matrix[num_rows_padded, num_cols_padded %/% 2 + 1] rfft2_cov =
    gp_periodic_exp_quad_cov_rfft2(num_rows_padded, num_cols_padded,
    sigma_plot, [lengthscale_plot, lengthscale_plot]',
    [num_rows_padded, num_cols_padded]');
  matrix[num_rows_padded, num_cols_padded] f = gp_inv_rfft2(
    z, rep_matrix(0, num_rows_padded, num_cols_padded), rfft2_cov);
  for(i in 1:num_plots) plot_effect[i] = f[plot_row[i],plot_col[i]];
  
  // time effect
  diagSPD_f = diagSPD_periodic(sigma_f, lengthscale_f, M_f);
  PHI_f = PHI_periodic(num_ratings_per_entry, M_f, 2*pi()/period, xn);
  for(i in 1:num_entries) time_effect[i] = intercept_f[i] + PHI_f * (diagSPD_f .* beta_f[i]);
  
  //rsm
  for (i in 1:num_raters) {
	tau_rater[i][1:(num_categories-2)] = tau_rater_free[i];
	tau_rater[i][num_categories-1] = -1*sum(tau_rater_free[i]);
  }
}

model {
  // params for plot effect
  sigma_plot ~ normal(0,4);
  to_vector(z) ~ std_normal();
  lengthscale_plot ~ inv_gamma(2,3);
  
  // params for time effect
  for(i in 1:num_entries) {
	  intercept_f[i] ~ normal(0, 4);	// Intercept of Gaussian Process
	  beta_f[i] ~ normal(0, 4);			// Hilbert Basis Coefficeints
  }
  lengthscale_f ~ inv_gamma(2,3);		// Gaussian Process lengthscale parameter
  sigma_f ~ inv_gamma(2,3);				// Gaussian Process variance parameter
  
  // params for rating model
  //for (i in 1:num_raters) tau_rater_free[i] ~ normal(0, 3);
  //for (i in 1:num_raters) rater_severity[i] ~ normal(0, 3);

// priors on rater severity and tau 
  for (i in 1:num_raters) target += normal_lpdf(rater_severity[i] | 0, 3);
  for (i in 1:num_raters) target += normal_lpdf(tau_rater[i] | 0, 2);
  // Modelling the target (y[n])
  for (n in 1:N) target += rsm(y[n], plot_effect[plot_id[n]]+time_effect[entry_id[n]][entry_cumcount[n]], rater_severity[rater_id[n]], tau_rater[rater_id[n]]);
}

// Predictions and log likelihood
generated quantities {
	array[num_entries] vector[pred_N] pred_time_effect;
	matrix[pred_N ,2*M_f] pred_PHI_f;
	vector[N] log_lik;
	pred_PHI_f = PHI_periodic(pred_N, M_f, 2*pi()/period, pred_xn);
	for(i in 1:num_entries) pred_time_effect[i] = intercept_f[i] + pred_PHI_f * (diagSPD_f .* beta_f[i]);
	for (n in 1:N) log_lik[n] = rsm(y[n], plot_effect[plot_id[n]]+time_effect[entry_id[n]][entry_cumcount[n]], rater_severity[rater_id[n]], tau_rater[rater_id[n]]);
}