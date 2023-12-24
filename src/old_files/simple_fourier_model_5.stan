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

  int<lower=1> num_rows;         // number of rows of the turf plot grid
  int<lower=1> num_cols;         // number of cols of the turf plot grid
  int<lower=1> num_rows_padded;	 // number of rows + padding
  int<lower=1> num_cols_padded;  // number of columns + padding  
  array[P] int<lower=1> plot_row;			 // row of the plot
  array[P] int<lower=1> plot_col;			 // column of the plot
}
parameters {
  vector[I-1] beta_free;
  vector[M-1] tau_free;
  vector[J] entry;
  // vector[P] eta;                      // normal(0, 1) trick
  real<lower=0> sigma;
  // real<lower=0> sigma_e;			 // additional base variance of plots
  // real<lower=0> inv_rho;			 // inv_lengthscale of the gp
  real<lower=0> alpha;				 // variance of the exponentiated quadratic GP
  // real<lower=log(2), upper=log(28)>  log_length_scale;		 // log rho
  matrix[num_rows_padded, num_cols_padded] z; // standard normal distribution
  real mu;
  real <lower=0>length_scale;
}
transformed parameters {
  vector[N] theta;                   // adjusted turf quality
  vector[I] beta;                    // rater severity
  vector[M] tau;                     // Rasch-Andrich threshold
  vector[P] plot;                    // plot effect
  // real<lower=0> length_scale = exp(log_length_scale);
  // create kernel for GP() based on plot distance
  // {
  // matrix[P, P] KERNEL;
  // real sq_sigma_e = square(sigma_e);
  // create kernel for GP() using distance matrix
  // for(i in 1:(P-1)){
  // for(j in (i+1):P){
  //    KERNEL[i,j] = square(alpha) * exp(-0.5*square(DIST[i,j] * inv_rho));
  //    KERNEL[j,i] = KERNEL[i,j];
  //  }
  // }
  // for(i in 1:P) KERNEL[i,i]= square(alpha) + sq_sigma_e;
  //  matrix[P, P] L_KERN = cholesky_decompose(KERNEL);
  //  plot = L_KERN * eta;
  //}
  // fourier method
  matrix[num_rows_padded, num_cols_padded %/% 2 + 1] rfft2_cov =
    gp_periodic_exp_quad_cov_rfft2(num_rows_padded, num_cols_padded,
    alpha, [length_scale, length_scale]',
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
}

model {
  // tau_free ~ normal(0,1);
  // beta_free ~ normal(0,1);
  sigma ~ student_t(3,0,1);
  entry ~ normal(0, sigma);
  // sigma_e ~ student_t(3,0,1);
  // eta ~ normal(0, 1);
  alpha ~ student_t(3,0,1);
  // inv_rho ~ gamma(5, 5);
  length_scale ~ student_t(3,0,1);
  to_vector(z) ~ std_normal();
  mu ~ student_t(2, 0, 1);
  target += normal_lpdf(beta | 0, 2);
  target += normal_lpdf(tau | 0, 2);
  for (n in 1:N){
	target += rsm(y[n], entry[jj[n]]+plot[pp[n]], beta[ii[n]], tau);
  }
}
