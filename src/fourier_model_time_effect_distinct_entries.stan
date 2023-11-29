functions {
// define a function to convert ratings onto a probability scale
// theta: turf quality at the time of rating
// beta: rater severity
// tau: rating thresholds
  #include gptools/util.stan
  #include gptools/fft.stan
  real rsm(int y, real theta, real beta, vector tau) {
    vector[rows(tau) + 1] unsummed;
    vector[rows(tau) + 1] probs;
    unsummed = append_row(rep_vector(0, 1), theta - beta - tau);
    probs = softmax(cumulative_sum(unsummed));
    return categorical_lpmf(y+1| probs);
  }
  // Hilbert basis methods helper functions
  vector diagSPD_EQ(real alpha, real rho, real L, int M) {	// Spectral densities corresponding to exponentiated quadratic kernel
	return alpha * sqrt(sqrt(2*pi()) * rho) * exp(-0.25*(rho*pi()/2/L)^2 * linspaced_vector(M, 1, M)^2);
  }
  matrix PHI(int N, int M, real L, vector x) { // Eigenfunctions of the Hilbert Basis on [-L,L]
    return sin(diag_post_multiply(rep_matrix(pi()/(2*L) * (x+L), M), linspaced_vector(M, 1, M)))/sqrt(L);
  }
}
data {
  int<lower=1> N;                // total number of responses
  int<lower=1> I;                // total number of ratings conducted
  int<lower=1> J;                // total number of entries
  int<lower=1> P;                // total number of plots
  int<lower=1> M;                // number of categories
  array[N] int<lower=1,upper=I> ii;    // rating id for y[n], rating id is defined by rater + date
  array[N] int<lower=1,upper=J> jj;    // entry of y[n]
  array[N] int<lower=1,upper=P> pp;    // plot id for y[n]
  array[N] int<lower=0> y;             // response
  matrix[P, P] DIST;             // distance matrix for all entries

// new data needed for fourier method
  int<lower=1> ratings_per_entry;
  int<lower=1> num_rows;         // number of rows of the turf plot grid
  int<lower=1> num_cols;         // number of cols of the turf plot grid
  int<lower=1> num_rows_padded;	 // number of rows + padding
  int<lower=1> num_cols_padded;  // number of columns + padding  
  array[P] int<lower=1> plot_row;			 // row of the plot
  array[P] int<lower=1> plot_col;			 // column of the plot
  
// new data for time element

  vector[N] day;	// day of year the rating was taken at
  real<lower=0> c_f;   // factor c to determine the boundary value L
  int<lower=1> M_f;    // number of basis functions
  array[N] int<lower=1> entry_cumcount;   
}
parameters {
  vector[I-1] beta_free;			 // used to compute the beta value in the rsm value
  vector[M-1] tau_free;				 // used to compute the tau value in rsm function 
  vector[J] entry;					 // individual plot entry
  
  // Area effect GP
  
  real<lower=0> sigma;				 // standard deviation of each entry
  real<lower=0> alpha_plot;			 // variance of the exponentiated quadratic GP for plots
  matrix[num_rows_padded, num_cols_padded] z; // standard normal distribution
  real mu;							 // mean of GP for plots
  real <lower=0>length_scale;		 // length scale of plot
  
  //
  
  vector[ratings_per_entry] z_time;					// standard normal distribution
  
  array[J] real<lower=0> sigma_time;			// standard deviation for time GP
  array[J] real intercept_f;
  array[J] vector[M_f] beta_f;          		// the basis functions coefficients
  array[J] real<lower=0> lengthscale_f; 		// lengthscale of f
  array[J] real<lower=0> sigma_f;       		// scale of f
}
transformed parameters {
  vector[N] theta;                   // adjusted turf quality
  vector[I] beta;                    // rater severity
  vector[M] tau;                     // Rasch-Andrich threshold
  vector[P] plot;                    // plot effect
  array[J] vector[ratings_per_entry] time_effect;			 // time effect
  // fourier method for plot
  matrix[num_rows_padded, num_cols_padded %/% 2 + 1] rfft2_cov =
    gp_periodic_exp_quad_cov_rfft2(num_rows_padded, num_cols_padded,
    alpha_plot, [length_scale, length_scale]',
    [num_rows_padded, num_cols_padded]');
  matrix[num_rows_padded, num_cols_padded] f = gp_inv_rfft2(
    z, rep_matrix(mu, num_rows_padded, num_cols_padded), rfft2_cov);
  for(i in 1:P) {
    plot[i] = f[plot_row[i],plot_col[i]];
  }
  beta[1:(I-1)] = beta_free;
  beta[I] = -1*sum(beta_free);
  tau[1:(M-1)] = tau_free;
  tau[M] = -1*sum(tau_free);
  
  real mean_day = mean(day);
  real sd_day = sd(day);
  
  // time effect
  array[J] real L_f;
  array[J] vector[M_f] diagSPD_f;
  array[J] matrix[ratings_per_entry ,M_f] PHI_f;
  //TODO: Initialize xn; array of dates corresponding to each entry
  // define dates_per_entry
  array[J] vector[ratings_per_entry] xn; // = (day - mean(day))/sd(day);			// standardized day variable
  for (n in 1:N) {
	xn[jj[n]][entry_cumcount[n]] = (day[n]-mean_day)/sd_day;
  }
  for(i in 1:J) {
	L_f[i] = c_f*max(xn[i]);
	diagSPD_f[i] = diagSPD_EQ(sigma_f[i], lengthscale_f[i], L_f[i], M_f);
	PHI_f[i] = PHI(ratings_per_entry, M_f, L_f[i], xn[i]);		// hilbert basis functions
	time_effect[i] = intercept_f[i] + PHI_f[i] * (diagSPD_f[i] .* beta_f[i]) + sigma_time[i]*z_time[i];
  }
}

model {
  sigma ~ student_t(3,0,1);
  entry ~ normal(0, sigma);

  alpha_plot ~ student_t(3,0,1);
  to_vector(z) ~ std_normal();
  mu ~ student_t(2, 0, 1);
  // each of these is an array
  for(i in 1:J) {
	  z_time[i] ~ normal(0,1);
	  sigma_time[i] ~ normal(0, 1);
	  intercept_f[i] ~ normal(0, 0.1);
	  beta_f[i] ~ normal(0, 1);
	  lengthscale_f[i] ~ normal(0, 1);
	  sigma_f[i] ~ normal(0, 1);
  }
  target += normal_lpdf(beta | 0, 2);
  target += normal_lpdf(tau | 0, 2);
  for (n in 1:N){
	target += rsm(y[n], entry[jj[n]]+plot[pp[n]]+time_effect[jj[n]][entry_cumcount[n]], beta[ii[n]], tau);
  }
}