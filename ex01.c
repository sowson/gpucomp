#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "opencl.h"
#include "ex01.h"

int ex01_N = 1;

cl_program* ex01_sum_kernel_program;
cl_kernel* ex01_sum_kernel;

int* ex01_gen_randoms;
cl_mem_ext ex01_cl_gen_randoms;
int* ex01_results;
cl_mem_ext ex01_cl_results;

clock_t ex01_benchmark_start()
{
    clock_t t;
    t = clock();
    return t;
}

void ex01_benchmark_stop(clock_t t, const char* func_name)
{
    t = clock() - t;
    double time_taken = ((double)t);
    printf("Exec of %s took %d ticks", func_name, (int)time_taken);
}

int* ex01_gen_rand(size_t n)
{
    int* set = calloc(n, sizeof(int));
    int i = 0;
    int v = 0;
    for(i = 0; i < n; ++i){
        v = (int)(rand() % 4 + 1);
        set[i] = v;
    }
    return set;
}

static const char* const ex01_sum_kernel_source = CONVERT_KERNEL_TO_STRING(

        inline void sum(__global int* a, int v)
        {
            int s = +1 * v;
            int n = 0;
            int o = 0;
            do {
                n = s + atom_xchg(a, o);
                s = o + atom_xchg(a, n);
            }
            while (s != o);
        }

        __kernel void sum_kernel(__global int *set, __global int* out)
        {
            int i = (get_group_id(0) + get_group_id(1)*get_num_groups(0)) * get_local_size(0) + get_local_id(0);

            sum(out, set[i]);
        }

);

void ex01_sum_kernel_init(int gpui) {
    // Inint OpenCL
    int* gpus = calloc(1, sizeof(int));
    gpus[0] = gpui;
    int ngpu = 1;
    opencl_init(gpus, ngpu);
    free(gpus);
    // Init GPU Computing
    int d;
    for (d = 0; d < ngpu; ++d) {
        if (d == 0) {
            ex01_sum_kernel_program = calloc(ngpu, sizeof(cl_program));
            ex01_sum_kernel = calloc(ngpu, sizeof(cl_kernel));
        }
        opencl_load_buffer(ex01_sum_kernel_source, strlen(ex01_sum_kernel_source),
                           &ex01_sum_kernel_program[d], ngpu, d);
        opencl_create_kernel(&ex01_sum_kernel_program[d], "sum_kernel",
                             &ex01_sum_kernel[d]);
    }
    // Init Mem
    ex01_gen_randoms = ex01_gen_rand(ex01_N);
    ex01_cl_gen_randoms = opencl_make_array(ex01_gen_randoms, sizeof(int), ex01_N);
    ex01_results = calloc(1, sizeof(int));
    ex01_cl_results = opencl_make_array(ex01_results, sizeof(int), 1);
}

void ex01_sum_kernel_deinit(int gpui) {
    int* gpus = calloc(1, sizeof(int));
    gpus[0] = gpui;
    int ngpu = 1;
    // DeInit Mem
    opencl_free(ex01_cl_results);
    opencl_free(ex01_cl_gen_randoms);
    // DeInit GPU Computing
    int d;
    for (d = 0; d < ngpu; ++d) {
        clReleaseKernel(ex01_sum_kernel[d]);
        ex01_sum_kernel[d] = 0;
        clReleaseProgram(ex01_sum_kernel_program[d]);
        ex01_sum_kernel_program[d] = 0;
        if (d == ngpu - 1) {
            free(ex01_sum_kernel_program);
            free(ex01_sum_kernel);
        }
    }
    // DeInint OpenCL
    opencl_deinit(gpus, ngpu);
    free(gpus);
}

int ex01_sum_cpu()
{
    int sum = 0;

    const int* set = ex01_gen_randoms;

    clock_t t = ex01_benchmark_start();
    int i = 0;
    for(i = 0; i < ex01_N; ++i) {
        sum += set[i];
    }
    ex01_benchmark_stop(t, "sum_cpu");
    printf(" with result %i.\n", sum);

    return sum;
}

int ex01_sum_gpu(int gpui)
{
    int sum = 0;

    ex01_results[0] = sum;
    opencl_push_array(ex01_cl_gen_randoms, ex01_gen_randoms);
    opencl_push_array(ex01_cl_results, ex01_results);

    dim2 dimGrid = opencl_gridsize(ex01_N);

    clock_t t = ex01_benchmark_start();
    opencl_kernel(ex01_sum_kernel[0], dimGrid, 4,
                  &ex01_cl_gen_randoms.org, sizeof(cl_mem),
                  &ex01_cl_results.org, sizeof(cl_mem)
    );
    ex01_benchmark_stop(t, "sum_gpu");

    opencl_pull_array(ex01_cl_results, ex01_results);
    sum = ex01_results[0];

    printf(" with result %i.\n", sum);

    return sum;
}

void ex01(int n, int t, int gpui) {
    ex01_N = n;

    ex01_sum_kernel_init(gpui);

    printf("Calculate SUM of %i randoms:\n", ex01_N);

    int i;
    for(i = 0; i < t; ++i)
    {
        printf("Compare %i:\n", i+1);
        ex01_sum_cpu();
        ex01_sum_gpu(gpui);
    }

    ex01_sum_kernel_deinit(gpui);
}
