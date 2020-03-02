data {
  int N;
  int N_d[N];
}


parameters {
  // parameters
  real<lower=0> e;
  real<lower=0> lambda;
  real N_I;
}


model {
  // Priors
  e ~ beta(1.1, 1.1);
  lambda ~ normal(0, 1);
  
  // Likelihood
  N_I ~ poisson(lambda);
  N_d ~ binomial(N_I, e);
}




