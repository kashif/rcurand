#include <Rcpp.h>
#include <cuda.h>
#include <curand.h>

using namespace Rcpp;

class Rcurand {
public:
  Rcurand(int rng_type = 100)
  {
    curandCreateGenerator(&gen_, (curandRngType_t)rng_type);
  }

  int setPseudoRandomGeneratorSeed(unsigned long seed)
  {
    return curandSetPseudoRandomGeneratorSeed(gen_, seed);
  }

  NumericVector generateUniform(size_t num)
  {
    NumericVector hostData(num);

    cudaMalloc((void **)&devData_, num*sizeof(double));
    curandGenerateUniformDouble(gen_, devData_, num);
    cudaMemcpy(hostData.begin() , devData_ , num * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(devData_);

    return hostData;
  }

  NumericVector generateNormal(size_t num, float mean=0.0f, float stddev=1.0f)
  {
    NumericVector hostData(num);

    cudaMalloc((void **)&devData_, num*sizeof(double));
    curandGenerateNormalDouble(gen_, devData_, num, mean, stddev);
    cudaMemcpy(hostData.begin() , devData_ , num * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(devData_);

    return hostData;
  }

  NumericVector generateLogNormal(size_t num, float mean=0.0f, float stddev=1.0f)
  {
    NumericVector hostData(num);

    cudaMalloc((void **)&devData_, num*sizeof(double));
    curandGenerateLogNormalDouble(gen_, devData_, num, mean, stddev);
    cudaMemcpy(hostData.begin() , devData_ , num * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(devData_);

    return hostData;
  }

  ~Rcurand()
  {
    curandDestroyGenerator(gen_);
  }

private:
  curandGenerator_t gen_;
  double *devData_;
};

