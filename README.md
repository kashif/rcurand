# Rcurand

The Rcurand package provides R bindings to NVIDIA's CURAND GPU random number host API via Rcpp.

## Prerequisites

You need to have the latest [CUDA release](https://developer.nvidia.com/cuda-downloads) installed on your system with a suitable GPU.

## Installation

Since this package is not currently available on CRAN, the easiest way to try it out is to get it via `devtools`:

```S
> install.packages("devtools")
> devtools::install_github("rcurand", "kashif")
```

## Usage

We can then load the package and generate random numbers:

```S
> library(Rcurand)
> curand <- new(Rcurand, 100)
> curand$setPseudoRandomGeneratorSeed(1234)
[1] 0
> curand$generateNormal(20, 30.3, 2.0)
 [1] 31.01748 27.82081 30.11701 29.53976 29.01008 30.23948 29.61954 27.30938 31.07547 32.31911
[11] 31.90599 28.58359 33.01653 32.58759 32.17076 33.93873 28.52211 31.63219 33.89073 30.49847
```

## The MIT License (MIT)

Copyright (c) 2013 Kashif Rasul <kashif.rasul@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.