#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "opencl.h"

int ex05_N;
cl_program* ex05_div_kernel_program;
cl_kernel* ex05_div_kernel;
float* ex05_gen_randoms;
cl_mem_ext ex05_cl_gen_randoms;
float* ex05_results;
cl_mem_ext ex05_cl_results;
clock_t ex05_benchmark_start();
void ex05_benchmark_stop(clock_t t, const char* func_name);
float* ex05_gen_rand(size_t n);
static const char* const ex05_div_kernel_source;
void ex05_div_kernel_init(int gpui);
void ex05_div_kernel_deinit(int gpui);
float ex05_div_cpu();
float ex05_div_gpu(int gpui);
void ex05(int n, int t, int gpui);