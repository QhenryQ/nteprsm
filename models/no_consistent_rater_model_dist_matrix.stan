functions {
// define a function to convert ratings onto a probability scale
// theta: turf quality at the time of rating
// beta: rater severity
// tau: rating thresholds
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
}
parameters {
  vector[I-1] beta_free;
  vector[M-1] tau_free;
  vector[J] entry;
  vector[P] eta;                      // normal(0, 1) trick
  real<lower=0> sigma;
  real<lower=0> sigma_e;
  real<lower=0> inv_rho;
  real<lower=0> alpha;
}
transformed parameters {
  vector[N] theta;                   // adjusted turf quality 
  vector[I] beta;                    // rater severity 
  vector[M] tau;                     // Rasch-Andrich threshold
  vector[P] plot;                    // plot effect
  // create kernel for GP() based on plot distance
  {
  matrix[P, P] KERNEL;
  real sq_sigma_e = square(sigma_e);
  // create kernel for GP() using distance matrix 
  for(i in 1:(P-1)){
   for(j in (i+1):P){
    KERNEL[i,j] = square(alpha) * exp(-0.5*square(DIST[i,j] * inv_rho));
    KERNEL[j,i] = KERNEL[i,j];
    }
  }
  for(i in 1:P) KERNEL[i,i]= square(alpha) + sq_sigma_e;     
  matrix[P, P] L_KERN = cholesky_decompose(KERNEL);
  plot = L_KERN * eta;
  }
  
  beta[1:(I-1)] = beta_free;
  beta[I] = -1*sum(beta_free);
  tau[1:(M-1)] = tau_free;
  tau[M] = -1*sum(tau_free);

  for (n in 1:N){
  theta[n] = entry[jj[n]] + plot[pp[n]]; 
  // turf quality at rating is partitioned into entry inherit part 
  // + plot location effect where plot location effect is model using a GP()
  }

}

model {
  target += normal_lpdf(beta | 0, 2);
  target += normal_lpdf(tau | 0, 2);
  entry ~ normal(0, sigma);
  sigma ~ student_t(3,0,1);
  sigma_e ~ student_t(3,0,1);
  eta ~ normal(0, 1);
  alpha ~ student_t(3,0,1);
  inv_rho ~ gamma(5, 5); 
  
  for (n in 1:N){
  target += rsm(y[n], theta[n], beta[ii[n]], tau);
  }
}
