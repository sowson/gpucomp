#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "opencl.h"

int ex02_N;
cl_program* ex02_sum_kernel_program;
cl_kernel* ex02_sum_kernel;
float* ex02_gen_randoms;
cl_mem_ext ex02_cl_gen_randoms;
float* ex02_results;
cl_mem_ext ex02_cl_results;
clock_t ex02_benchmark_start();
void ex02_benchmark_stop(clock_t t, const char* func_name);
float* ex02_gen_rand(size_t n);
static const char* const ex02_sum_kernel_source;
void ex02_sum_kernel_init(int gpui);
void ex02_sum_kernel_deinit(int gpui);
float ex02_sum_cpu();
float ex02_sum_gpu(int gpui);
void ex02(int n, int t, int gpui);
