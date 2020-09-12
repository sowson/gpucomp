#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "opencl.h"

int ex04_N ;
cl_program* ex04_mul_kernel_program;
cl_kernel* ex04_mul_kernel;
float* ex04_gen_randoms;
cl_mem_ext ex04_cl_gen_randoms;
float* ex04_results;
cl_mem_ext ex04_cl_results;
clock_t ex04_benchmark_start();
void ex04_benchmark_stop(clock_t t, const char* func_name);
float* ex04_gen_rand(size_t n);
static const char* const ex04_mul_kernel_source;
void ex04_mul_kernel_init(int gpui);
void ex04_mul_kernel_deinit(int gpui);
float ex04_mul_cpu();
float ex04_mul_gpu(int gpui);
void ex04(int n, int t, int gpui);