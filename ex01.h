#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "opencl.h"

int ex01_N;

cl_program* ex01_sum_kernel_program;
cl_kernel* ex01_sum_kernel;

int* ex01_gen_randoms;
cl_mem_ext ex01_cl_gen_randoms;
int* ex01_results;
cl_mem_ext ex01_cl_results;

clock_t ex01_benchmark_start();
void ex01_benchmark_stop(clock_t t, const char* func_name);
int* ex01_gen_rand(size_t n);
static const char* const ex01_sum_kernel_source;
void ex01_sum_kernel_init(int gpui);
void ex01_sum_kernel_deinit(int gpui);
int ex01_sum_cpu();
int ex01_sum_gpu(int gpui);
void ex01(int n, int t, int gpui);
