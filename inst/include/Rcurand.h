#include <Rcpp.h>
#include <cuda.h>
#include <curand.h>

using namespace Rcpp;
using namespace std;

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
    vector<float> hostData(num);

    cudaMalloc((void **)&devData_, num*sizeof(float));
    curandGenerateUniform(gen_, devData_, num);
    cudaMemcpy(&(hostData[0]), devData_, num * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(devData_);

    return wrap(hostData);
  }

  NumericVector generateNormal(size_t num, float mean=0.0f, float stddev=1.0f)
  {
    vector<float> hostData(num);

    cudaMalloc((void **)&devData_, num*sizeof(float));
    curandGenerateNormal(gen_, devData_, num, mean, stddev);
    cudaMemcpy(&(hostData[0]), devData_, num * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(devData_);

    return wrap(hostData);
  }

  NumericVector generateLogNormal(size_t num, float mean=0.0f, float stddev=1.0f)
  {
    vector<float> hostData(num);

    cudaMalloc((void **)&devData_, num*sizeof(float));
    curandGenerateLogNormal(gen_, devData_, num, mean, stddev);
    cudaMemcpy(&(hostData[0]), devData_, num * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(devData_);

    return wrap(hostData);
  }

  NumericVector generatePoisson(size_t num, float lambda)
  {
    vector<unsigned int> hostData(num);
    unsigned int *devData;

    cudaMalloc((void **)&devData, num*sizeof(unsigned int));
    curandGeneratePoisson(gen_, devData, num, (double)lambda);
    cudaMemcpy(&(hostData[0]), devData, num * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaFree(devData);

    return wrap(hostData);
  }

  ~Rcurand()
  {
    curandDestroyGenerator(gen_);
  }

private:
  curandGenerator_t gen_;
  float *devData_;
};

