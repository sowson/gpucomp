#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "opencl.h"

int ex03_N;
cl_program* ex03_sub_kernel_program;
cl_kernel* ex03_sub_kernel;
float* ex03_gen_randoms;
cl_mem_ext ex03_cl_gen_randoms;
float* ex03_results;
cl_mem_ext ex03_cl_results;
clock_t ex03_benchmark_start();
void ex03_benchmark_stop(clock_t t, const char* func_name);
float* ex03_gen_rand(size_t n);
static const char* const ex03_sub_kernel_source;
void ex03_sub_kernel_init(int gpui);
void ex03_sub_kernel_deinit(int gpui);
float ex03_sub_cpu();
float ex03_sub_gpu(int gpui);
void ex03(int n, int t, int gpui);