functions {
  #include gptools/util.stan
  #include gptools/fft.stan
// Converts ratings onto a probability scale using the Rasch Rating Scale model
// theta: turfgrass quality at the time of rating
// tau: rating thresholds
  real rsm(int y, real theta, vector tau) {
    vector[rows(tau) + 1] unsummed;
    vector[rows(tau) + 1] probs;
    unsummed = append_row(rep_vector(0, 1), theta - tau);
    probs = softmax(cumulative_sum(unsummed));
    return categorical_lpmf(y+1| probs);
  }
  
/*   // Hilbert basis methods helper functions
  
  // Returns the spectral densities corresponding to an exponentiated quadratic (RBF) kernel
  // alpha 	: The variance of the kernel used
  // rho 	: The lengthscale of the kernel
  // L		: The Hilbert Basis is over the interval [-L,L]
  // M 		: Number of Hilbert basis functions
  vector diagSPD_EQ(real alpha, real rho, real L, int M) {	// Spectral densities corresponding to exponentiated quadratic kernel
	return alpha * sqrt(sqrt(2*pi()) * rho) * exp(-0.25*(rho*pi()/2/L)^2 * linspaced_vector(M, 1, M)^2);
  }
  
  // Returns the evaluations of the Eigenfunctions of the Hilbert Basis on [-L,L]
  // N : Dimension of the vector x 
  // M : number of basis functions
  // L : Hilbert basis is over the interval [-L,L]
  // x : The values we want to take evaluate the Hilbert Basis on
  matrix PHI(int N, int M, real L, vector x) { // Eigenfunctions of the Hilbert Basis on [-L,L]
    return sin(diag_post_multiply(rep_matrix(pi()/(2*L) * (x+L), M), linspaced_vector(M, 1, M)))/sqrt(L);
  }
  
  // For Periodic Kernels
  vector diagSPD_periodic(real alpha, real rho, int M) {
    real a = 1/rho^2;
    vector[M] q = exp(log(alpha) + 0.5 * (log(2) - a + to_vector(log_modified_bessel_first_kind(linspaced_int_array(M, 1, M), a))));
    return append_row(q,q);
  }
  matrix PHI_periodic(int N, int M, real w0, vector x) {
    matrix[N,M] mw0x = diag_post_multiply(rep_matrix(w0*x, M), linspaced_vector(M, 1, M));
    return append_col(cos(mw0x), sin(mw0x));
  } */
  
}

data {
  int<lower=1> N;                			// total number of responses
  int<lower=1> num_raters;                  // total number of distinct raters
  int<lower=1> num_unique_days;			    // total number of days
  int<lower=1> num_entries;                	// total number of entries
  int<lower=1> num_plots;                	// total number of plots
  int<lower=1> num_categories;              // number of categories
  array[N] int<lower=1,upper=num_raters> rater_id;    	// rater of y[n]
  array[N] int<lower=1,upper=num_unique_days> day_id; // rating day id of y[n]
  array[N] int<lower=1,upper=num_entries> entry_id;    	// entry of y[n]
  array[N] int<lower=1,upper=num_plots> plot_id;    	// plot id for y[n]
  array[N] int<lower=1,upper=num_categories> y;         // QUALITY variable for y[n]
  
// data needed for fourier method
  int<lower=1> num_rows;         		// number of rows of the turf plot grid
  int<lower=1> num_cols;         		// number of cols of the turf plot grid
  int<lower=1> num_rows_padded;	 		// number of rows + padding
  int<lower=1> num_cols_padded;  		// number of columns + padding  
  array[num_plots] int<lower=1> plot_row;			 // row of the plot corresponding to plot_id
  array[num_plots] int<lower=1> plot_col;			 // column of the plot corresponding to plot_id
  
// new data for time element
  array[N] real time;	// time of year the rating was taken at
  array[num_unique_days] real day_id_to_time;
// data for predictions/generated quantitites
  int<lower=1> pred_N;        // total number of generated responses
  vector[pred_N] pred_time;	// day of year the generated rating was taken at
}
parameters {
  // Area effect GPs
  real<lower=0> alpha_plot;			 // variance of the exponentiated quadratic GP for plots
  matrix[num_rows_padded, num_cols_padded] z; // standard normal distribution
  real <lower=0>lengthscale_plot;		 // length scale of plot
  
  // time GP
  vector[num_entries] z_time;					// standard normal distribution
  array[num_entries] real intercept_f;
  array[num_entries] real<lower=0> lengthscale_f; 		// lengthscale of f
  array[num_entries] real<lower=0> sigma_f;       		// scale(variance) of f
  array[num_entries] vector[num_unique_days] time_effect_eta;			 // time effect
  vector[num_unique_days] jitter;			 		 // time effect
  // rater parameters
  array[num_raters] vector[num_categories-1] tau_rsm_free;		 // used to compute the tau value in rsm function 
  // real <lower=0> smoothing_factor;
}

transformed parameters {
  vector[num_plots] plot_effect;     // plot effect
  array[num_entries] vector[num_unique_days] time_effect;			 // time effect
  array[num_entries] matrix[num_unique_days, num_unique_days] time_kernel;
  array[num_raters] vector[num_categories] tau_rsm;    					 // Rasch-Andrich threshold
  
  // fourier method for plot GP
  matrix[num_rows_padded, num_cols_padded %/% 2 + 1] rfft2_cov =
    gp_periodic_exp_quad_cov_rfft2(num_rows_padded, num_cols_padded,
    alpha_plot, [lengthscale_plot, lengthscale_plot]',
    [num_rows_padded, num_cols_padded]');
  matrix[num_rows_padded, num_cols_padded] f = gp_inv_rfft2(
    z, rep_matrix(0, num_rows_padded, num_cols_padded), rfft2_cov);
  for(i in 1:num_plots) {
    plot_effect[i] = f[plot_row[i],plot_col[i]];
  }
  
// time effect
  for(en in 1:num_entries) {
	for (i in 1:(num_unique_days)-1) {
	  for (j in (i+1):num_unique_days) {
		time_kernel[en][i,j] = sigma_f[en]^2 * exp(-2*(sin(pi()*abs(day_id_to_time[i]-day_id_to_time[j])))^2/(lengthscale_f[en])^2);
		time_kernel[en][j,i] = time_kernel[en][i,j];
	  }
	}
	for (i in 1:(num_unique_days)) {
		time_kernel[en][i,i] = square(sigma_f[en]) + square(jitter[i]);
	}
	time_effect[en] = cholesky_decompose(time_kernel[en]) * time_effect_eta[en] + intercept_f[en];
  }
  
  // parameters for RSM model
  // beta : Rater Severity
  // tau : Rating Thresholds
  for (i in 1:num_raters) {
	tau_rsm[i][1:(num_categories-1)] = tau_rsm_free[i];
	tau_rsm[i][num_categories] = -1*sum(tau_rsm_free[i]);
  }
}

model {
  // params for plot effect
  alpha_plot ~ student_t(3,0,1);
  to_vector(z) ~ std_normal();
  // params for time effect
  jitter ~ normal(0,0.0001);		// ensure matrix is Positive Definite
  for(i in 1:num_entries) {
	  intercept_f[i] ~ normal(0, 4);	// Intercept of Gaussian Process
	  lengthscale_f[i] ~ normal(0,1);	// Gaussian Process lengthscale parameter
	  sigma_f[i] ~ normal(0,4);			// Gaussian Process variance parameter
	  time_effect_eta[i] ~ normal(0,1); // for sampling
  }
  for (i in 1:num_raters) {
	target += normal_lpdf(tau_rsm[i] | 0, 2);
  }
  for (n in 1:N){
	target += rsm(y[n], plot_effect[plot_id[n]]+time_effect[entry_id[n]][day_id[n]], tau_rsm[rater_id[n]]);
  }
}

// Predictions
/* generated quantities {
	array[num_entries] vector[pred_N] pred_time_effect;			 // time effect
    vector[pred_N] pred_xn;
	array[num_entries] matrix[pred_N ,2*M_f] pred_PHI_f;
	for (n in 1:pred_N) {
		pred_xn[n] = (pred_time[n]-mean_time)/sd_time; 		// standardized day variable
	}
	for(i in 1:num_entries) {
		pred_PHI_f[i] = PHI_periodic(num_ratings_per_entry, M_f, 2*pi()*sd_time, pred_xn);
		pred_time_effect[i] = intercept_f[i] + pred_PHI_f[i] * (diagSPD_f[i] .* beta_f[i]);
    }
} */