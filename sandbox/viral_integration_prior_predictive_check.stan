functions {
  real p_theo(real N_d, real N_cfp, real N_I) {
    real P = N_cfp * N_I;
    return N_d * P;
  }
}

data {
  int N;
}


generated quantities {
  // parameters
  real<lower=0> e;
  real<lower=0> s;
  real<lower=0> lambda;
  int N_I;
  int N_d;
  int N_cfp;

  // data
  real p[N];
  
  e = beta_rng(1.1, 1.1);
  s = beta_rng(1.1, 1.1);
  lambda = normal_rng(0, 1);
  
  // Likelihood
  N_I = poisson_rng(lambda);
  N_d = binomial_rng(N_I, e);
  N_cfp = binomial_rng(N_I, 1-s);
  
  for (i in 1:N) {
    p[i] = p_theo(N_d, N_cfp, N_I);
  }
}



