#include "Rcurand.h"

RCPP_MODULE(Rcurand_module) {

  class_<Rcurand>("Rcurand")
  .constructor<int>("Create new random number generator of type rng_type.")

  .method("setPseudoRandomGeneratorSeed", &Rcurand::setPseudoRandomGeneratorSeed,
    "Set the seed value of the pseudo-random number generator.")
  .method("generateUniform", &Rcurand::generateUniform,
    "Generate n uniformly distributed doubles.")
  .method("generateNormal", &Rcurand::generateNormal,
    "Generate n normally distributed doubles with mean and stddev.")
  .method("generateLogNormal", &Rcurand::generateNormal,
    "Generate n log-normally distributed doubles with mean and stddev.")
  .method("generatePoisson", &Rcurand::generatePoisson,
    "Generate n Poisson-distributed integer values with given lambda.")
  ;
}
