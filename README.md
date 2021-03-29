PageRank (PR) algorithm for sparse graphs using CUDA Toolkit.

> NOTE: Execution time doesnt include memory allocation. It is done ahead of time.

```bash
0/17 tests failed.
Loading graph ...
order: 29008 size: 38416 {}
order: 29008 size: 38416 {}
[00885.7 ms] pageRankPush
[00009.5 ms] pageRank
[00020.1 ms] pageRankCuda


==1163== NVPROF is profiling process 1163, command: ./a.out data/aug2d.mtx
==1163== Profiling application: ./a.out data/aug2d.mtx
==1163== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   94.24%  16.860ms        25  674.42us  670.04us  683.00us  void pageRankKernelStep<float>(float*, float*, int*, int*, float, int)
                    1.83%  327.80us        25  13.112us  12.991us  14.016us  void sumIfNotKernel<float, int>(float*, float*, int*, int)
                    1.76%  314.01us        25  12.560us  12.448us  12.960us  void errorAbsKernel<float>(float*, float*, float*, int)
                    0.89%  158.37us        56  2.8270us  2.3670us  16.896us  [CUDA memcpy DtoH]
                    0.67%  119.29us        25  4.7710us  4.7030us  5.3120us  void multiplyKernel<float>(float*, float*, float*, int)
                    0.43%  76.127us        10  7.6120us  1.5360us  24.096us  [CUDA memcpy HtoD]
                    0.04%  7.4880us         1  7.4880us  7.4880us  7.4880us  void pageRankFactorKernel<float, int>(float*, int*, float, int)
                    0.04%  6.3360us         1  6.3360us  6.3360us  6.3360us  void dotProductKernel<int>(int*, int*, int*, int)
                    0.03%  6.1760us         1  6.1760us  6.1760us  6.1760us  void errorAbsKernel<int>(int*, int*, int*, int)
                    0.03%  5.6320us         1  5.6320us  5.6320us  5.6320us  void sumKernel<int>(int*, int*, int)
                    0.02%  4.4480us         1  4.4480us  4.4480us  4.4480us  void fillKernel<float>(float*, int, float)
                    0.02%  3.0080us         1  3.0080us  3.0080us  3.0080us  void addKernel<int>(int*, int, int)
                    0.01%  2.3360us         1  2.3360us  2.3360us  2.3360us  void fillKernel<int>(int*, int, int)
      API calls:   87.44%  153.74ms        19  8.0918ms  3.2640us  152.75ms  cudaMalloc
                   10.88%  19.137ms        66  289.95us  12.216us  723.58us  cudaMemcpy
                    0.83%  1.4652ms       107  13.693us  7.5780us  159.44us  cudaLaunchKernel
                    0.39%  690.48us        18  38.360us  3.8500us  152.74us  cudaFree
                    0.32%  562.27us         1  562.27us  562.27us  562.27us  cuDeviceTotalMem
                    0.11%  197.24us        97  2.0330us     156ns  74.846us  cuDeviceGetAttribute
                    0.01%  24.393us         1  24.393us  24.393us  24.393us  cuDeviceGetName
                    0.00%  8.0020us         1  8.0020us  8.0020us  8.0020us  cuDeviceGetPCIBusId
                    0.00%  1.9220us         3     640ns     154ns  1.0450us  cuDeviceGetCount
                    0.00%  1.7090us         2     854ns     286ns  1.4230us  cuDeviceGet
                    0.00%     271ns         1     271ns     271ns     271ns  cuDeviceGetUuid
```

<br>
<br>


## Usage

```bash
# Download program
rm -rf pagerank-sparse
git clone https://github.com/cudaf/pagerank-sparse
```

```bash
# Run
cd pagerank-sparse && nvcc -Xcompiler -fopenmp -O3 main.cu
cd pagerank-sparse && nvprof ./a.out data/aug2d.mtx
```

<br>
<br>


## References

- [PageRank Algorithm, Mining massive Datasets (CS246), Stanford University](http://snap.stanford.edu/class/cs246-videos-2019/lec9_190205-cs246-720.mp4)
- [CUDA by Example :: Jason Sanders, Edward Kandrot](http://www.mat.unimi.it/users/sansotte/cuda/CUDA_by_Example.pdf)
- [RAPIDS CUDA DataFrame Internals for C++ Developers - S91043](https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s91043-rapids-cuda-dataframe-internals-for-c++-developers.pdf)
- [lower_bound == upper_bound](https://stackoverflow.com/a/12159150/1413259)
- [Switch statement: must default be the last case?](https://stackoverflow.com/a/3110163/1413259)
- [Equivalent of “using namespace X” for scoped enumerations?](https://stackoverflow.com/a/9450358/1413259)
