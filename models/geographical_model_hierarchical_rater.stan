functions {
  #include gptools/util.stan
  #include gptools/fft.stan
// Converts ratings onto a probability scale using the Rasch Rating Scale model
// theta: turfgrass quality at the time of rating
// beta: rating thresholds
  real rsm(int y, real theta, vector beta) {
    vector[rows(beta) + 1] unsummed;
    vector[rows(beta) + 1] probs;
    unsummed = append_row(rep_vector(0, 1), theta - beta);
    probs = softmax(cumulative_sum(unsummed));
    return categorical_lpmf(y+1| probs);
  }
  
  // Hilbert basis methods helper functions
  
  // Returns the spectral density corresponding to a the kernel
  // alpha 	: The variance of the kernel used
  // rho 	: The lengthscale of the kernel
  // L 		: period of domain
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
  // L : period of domain
  // x : The values we want to take evaluate the Hilbert Basis on
  matrix PHI_periodic(int N, int M, real w0, vector x) {
    matrix[N,M] mw0x = diag_post_multiply(rep_matrix(w0*x, M), linspaced_vector(M, 1, M));
    return append_col(cos(mw0x), sin(mw0x));
  }
  
  matrix cov_exp_quad(matrix x, real alpha, real rho) {
    int N = rows(x);
    matrix[N, N] cov;
    for (i in 1:N)
      for (j in 1:N)
        cov[i, j] = alpha^2 * exp(-0.5 * squared_distance(x[i], x[j]) / rho^2);
    return cov;
  }
}

data {  
  // general dataset information
  int<lower=1> N;                			// total number of responses for rating data
  int<lower=1> L;							// total number of trial locations
  int<lower=2> num_categories;              // number of categories
  int<lower=1> num_entries;             // total number of entries
  array[N] int<lower=0,upper=num_categories> y;         // Rating quality, in process_data(), y is reindex from 0 to max, and max is the number of categories. 
  array[N] int trial_loc; // 1 corresponds to location A, 2 corresponds to location B
  
  // data needed for Plot Effect GP
  int padding; 
  
  array[N] int<lower=1> plot_id;     // plot id for response n
  
  // trial layout information
  array[L] int<lower=1> num_plots;           // total number of plots for each location
  int<lower=1> total_num_plots;  			 // total number of plots, should equal to (sum_num_plots)
  array[L] int<lower=1> num_rows;         		// number of rows for each locaton
  array[L] int<lower=1> num_cols;         		// number of cols for each locaton
  array[total_num_plots] int<lower=1> plot_row;			 // row of the plot corresponding to plot_id
  array[total_num_plots] int<lower=1> plot_col;			 // column of the plot corresponding to plot_id
  
  // geographical location
  vector[L] lats;
  vector[L] lons;
  
  // data needed for seasonality GP
  int<lower=1> M_f;    					// number of basis functions
  array[N] int<lower=1,upper=num_entries> entry_id;    	// entry of response n
  int<lower=1> num_rating_events;
  array[N] int<lower=1> rating_event_codes; // rating event of response n
  array[num_rating_events] real time;       // time of year corresponding to each rating event, float frmo 0-1
  
  // data needed for rater information
  int<lower=1> num_raters;                  // total number of distinct raters
  array[N] int<lower=1,upper=num_raters> rater_id;    	// rater of response n
  
  // data for predictions/generated quantitites
  int<lower=1> pred_N;        			// total number of generated time effect responses for each location
}

transformed data {
  // spatial effect for various trial lcoations
  array[L] int<lower=0> cumulative_num_plots;	 // cumsum of num_plots
  int <lower=0> cumulative_total = 0;
  for (l in 1:L) {
	  cumulative_num_plots[l] = cumulative_total;
	  cumulative_total = cumulative_total + num_plots[l];
  }
  print("Transforming data......");
  // Time Effect
  real mean_time = mean(time);
  real sd_time = sd(time);
  real period = 1/sd_time;
  //vector[pred_N] pred_xn_A;
  //vector[pred_N] pred_xn_B;
  real mean_lats = mean(lats);
  real mean_lons = mean(lons);
  real sd_lats_lons = (sd(lats) + sd(lons))/2;
  
  vector[L] normalized_lats = (lats-mean_lats)/sd_lats_lons;
  vector[L] normalized_lons = (lons-mean_lats)/sd_lats_lons;

  matrix[L, 2] x_geo;
  for (l in 1:L) {
    x_geo[l, 1] = normalized_lats[l];
    x_geo[l, 2] = normalized_lons[l];
  }
  
  int<lower=1> spatial_grid_rows = max(num_rows) + padding;
  int<lower=1> spatial_grid_cols = max(num_cols) + padding;
  
  // xn is standardized array of dates corresponding to each entry
  vector[num_rating_events] xn; 
  for (n in 1:num_rating_events) xn[n] = (time[n]-mean_time)/sd_time;
  matrix[num_rating_events ,2*M_f] PHI_f = PHI_periodic(num_rating_events, M_f, 2*pi()/period, xn);
  
    // for predictions
  vector[pred_N] pred_time;				// time of year the generated rating was taken at (Multiply by 365 to get day of year) 
  for (i in 1:pred_N) pred_time[i] = i * 1.0 / pred_N ; 
  vector[pred_N] pred_xn = (pred_time-mean_time)/sd_time;
  matrix[pred_N ,2*M_f] pred_PHI_f = PHI_periodic(pred_N, M_f, 2*pi()/period, pred_xn);
  print("Finished transforming data......");
}

parameters {
  // Spatial Effect Gaussian Process
  array[L] real<lower=0> sigma_plot;			 			// variance of the exponentiated quadratic GP for plots
  array[L] matrix[spatial_grid_rows, spatial_grid_cols] z; 	// standard normal distribution
  array[L] real <lower=0> lengthscale_plot;		 		// lengthscale of spatial effect
  
  // Time Effect Gaussian Process
  real<lower=0> lengthscale_f; 					// shared lengthscale of GP
  real<lower=0> sigma_f;       					// shared scale(variance) of GP
  
  // geographical parameter
  real<lower=0> lengthscale_geo;
  real<lower=0> sigma_geo;
  vector[L] z_geo;
  
  // geographical covariance
  matrix[L, num_entries] z_intercept;
  array[2*M_f] matrix[num_entries, L] z_beta; // [basis, location, entry]
  
  // Rater parameters
  vector[num_categories-1] beta_rater_population_mean;
  real<lower=0> beta_rater_population_var;
  array[num_raters] vector[num_categories-1] beta_rater;			// Category thresholds of each rater
}

transformed parameters {
  // print("Setting transformed parameters............");
  array[L] vector[max(num_plots)] plot_effect_loc;     							// plot effect
  array[N] real plot_effect;
  
    // Time Effect Gaussian Process
  array[L, num_entries] vector[num_rating_events] time_effect;	// time effect
  matrix[L, num_entries] intercept_f;			// intercept of the GP
  array[num_entries] matrix[2*M_f, L] beta_f;      // basis functions coefficients
  array[L] vector[2*M_f] diagSPD_f;										// spectral densities of periodic kernel

  matrix[L, L] geo_cov;
  matrix[L, L] L_cov;
	
  // fourier method for Plot Effect _A
  // print("Assigning spatial effect............");
  for (l in 1:L) {
	  matrix[spatial_grid_rows, spatial_grid_cols %/% 2 + 1] rfft2_cov =
		gp_periodic_exp_quad_cov_rfft2(spatial_grid_rows, spatial_grid_cols,
		sigma_plot[l], [lengthscale_plot[l], lengthscale_plot[l]]',
		[spatial_grid_rows, spatial_grid_cols]');
	  matrix[spatial_grid_rows, spatial_grid_cols] f = gp_inv_rfft2(
		z[l], rep_matrix(0, spatial_grid_rows, spatial_grid_cols), rfft2_cov);
	int plot_start = cumulative_num_plots[l];
	for (i in 1:num_plots[l]) plot_effect_loc[l][i] = f[plot_row[plot_start + i],plot_col[plot_start + i]];
  }
  // print("Assigning spatial effect 1:N vector............");
  for(i in 1:N) {
    plot_effect[i] = plot_effect_loc[trial_loc[i]][plot_id[i]];
  }
  
    
  // geogrpahical spatial coregionalization kernel
  geo_cov = cov_exp_quad(x_geo, sigma_geo, lengthscale_geo);

  for (l in 1:L) geo_cov[l, l] = geo_cov[l, l] + 1e-12;
  L_cov = cholesky_decompose(geo_cov);
  
  // print("Assigning intercept_f............");
  
  for (j in 1:num_entries) {
    intercept_f[, j] = L_cov * col(z_intercept, j);
	// print("resulting intercept-mul size: ", dims((L_cov * col(z_intercept, j))[2]));
  }
  // print("Testing print statement............");
  // beta_f: array[num_entries] matrix[L, 2*M_f]
  for (j in 1:num_entries)
    for (k in 1:(2*M_f))
	  {
		beta_f[j][k,] = row(z_beta[k], j) * L_cov;
	  }
	  
  // Hilbert Basis approximation for Time effect
  for (l in 1:L) {
	diagSPD_f[l] = diagSPD_periodic(sigma_f, lengthscale_f, M_f);
	for(i in 1:num_entries) time_effect[l,i] = intercept_f[l,i] + PHI_f * (diagSPD_f[l] .* (beta_f[i][,l]));
  }

}

model {
  // params for Plot Effect
  for (l in 1:L) {
	sigma_plot[l] ~ normal(0,1);
	lengthscale_plot[l] ~ inv_gamma(5,3);
	to_vector(z[l]) ~ std_normal();
  }
  
  // params for Time Effect
  to_vector(z_intercept) ~ normal(0, 1);
  for (k in 1:(2*M_f))
    to_vector(z_beta[k]) ~ normal(0, 1);
  lengthscale_f ~ inv_gamma(5,3);		// Gaussian Process lengthscale parameter
  sigma_f ~ normal(0,1);				// Gaussian Process variance parameter
  
  // params for geographical
  sigma_geo ~ normal(0,2);
  lengthscale_geo ~ inv_gamma(5,3);
  
  // priors on beta 
  beta_rater_population_var ~ inv_gamma(5,3);
  target += normal_lpdf(beta_rater_population_mean | 0, 2);
  for (i in 1:num_raters) target += normal_lpdf(beta_rater[i] | beta_rater_population_mean, beta_rater_population_var);
  
  // Modelling the target (y[n])
  for (n in 1:N) target += rsm(y[n], plot_effect[n]+time_effect[trial_loc[n], entry_id[n]][rating_event_codes[n]], beta_rater[rater_id[n]]);
}

// Predictions and log likelihood
generated quantities {
	array[L, num_entries] vector[pred_N] pred_time_effect;
	vector[N] log_lik;
	for (l in 1:L)
		for(i in 1:num_entries) pred_time_effect[l,i] = intercept_f[l,i] + pred_PHI_f * (diagSPD_f[l] .* (beta_f[i][,l]));
    
	for (n in 1:N) log_lik[n] = rsm(y[n], plot_effect[n]+time_effect[trial_loc[n], entry_id[n]][rating_event_codes[n]], beta_rater[rater_id[n]]);
} 