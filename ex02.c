#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "opencl.h"
#include "ex02.h"

int ex02_N = 1;

cl_program* ex02_sum_kernel_program;
cl_kernel* ex02_sum_kernel;

float* ex02_gen_randoms;
cl_mem_ext ex02_cl_gen_randoms;
float* ex02_results;
cl_mem_ext ex02_cl_results;

clock_t ex02_benchmark_start()
{
    clock_t t;
    t = clock();
    return t;
}

void ex02_benchmark_stop(clock_t t, const char* func_name)
{
    t = clock() - t;
    double time_taken = ((double)t);
    printf("Exec of %s took %d ticks", func_name, (int)time_taken);
}

float* ex02_gen_rand(size_t n)
{
    float* set = calloc(n, sizeof(float));
    int i = 0;
    float v = 0;
    for(i = 0; i < n; ++i){
        v = (float)(rand() % 4 + 1);
        set[i] = v;
    }
    return set;
}

static const char* const ex02_sum_kernel_source = CONVERT_KERNEL_TO_STRING(

        inline void sum(__global float* a, float v)
        {
            float s = +1.f * v;
            float n = 0.f;
            float o = 0.f;
            do {
                n = s + atom_xchg(a, o);
                s = o + atom_xchg(a, n);
            }
            while (s != o);
        }

        __kernel void sum_kernel(__global float *set, __global float* out)
        {
            int i = (get_group_id(0) + get_group_id(1)*get_num_groups(0)) * get_local_size(0) + get_local_id(0);

            sum(out, set[i]);
        }

);

void ex02_sum_kernel_init(int gpui) {
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
            ex02_sum_kernel_program = calloc(ngpu, sizeof(cl_program));
            ex02_sum_kernel = calloc(ngpu, sizeof(cl_kernel));
        }
        opencl_load_buffer(ex02_sum_kernel_source, strlen(ex02_sum_kernel_source),
                           &ex02_sum_kernel_program[d]);
        opencl_create_kernel(&ex02_sum_kernel_program[d], "sum_kernel",
                             &ex02_sum_kernel[d]);
    }
    // Init Mem
    ex02_gen_randoms = ex02_gen_rand(ex02_N);
    ex02_cl_gen_randoms = opencl_make_array(ex02_gen_randoms, sizeof(float), ex02_N);
    ex02_results = calloc(1, sizeof(float));
    ex02_cl_results = opencl_make_array(ex02_results, sizeof(float), 1);
}

void ex02_sum_kernel_deinit(int gpui) {
    int* gpus = calloc(1, sizeof(int));
    gpus[0] = gpui;
    int ngpu = 1;
    // DeInit Mem
    opencl_free(ex02_cl_results);
    opencl_free(ex02_cl_gen_randoms);
    // DeInit GPU Computing
    int d;
    for (d = 0; d < ngpu; ++d) {
        clReleaseKernel(ex02_sum_kernel[d]);
        ex02_sum_kernel[d] = 0;
        clReleaseProgram(ex02_sum_kernel_program[d]);
        ex02_sum_kernel_program[d] = 0;
        if (d == ngpu - 1) {
            free(ex02_sum_kernel_program);
            free(ex02_sum_kernel);
        }
    }
    // DeInint OpenCL
    opencl_deinit(gpus, ngpu);
    free(gpus);
}

float ex02_sum_cpu()
{
    float sum = 0;

    const float* set = ex02_gen_randoms;

    clock_t t = ex02_benchmark_start();
    int i = 0;
    for(i = 0; i < ex02_N; ++i) {
        sum += set[i];
    }
    ex02_benchmark_stop(t, "sum_cpu");
    printf(" with result %.2f.\n", sum);

    return sum;
}

float ex02_sum_gpu(int gpui)
{
    float sum = 0;

    ex02_results[0] = sum;
    opencl_push_array(ex02_cl_gen_randoms, ex02_gen_randoms);
    opencl_push_array(ex02_cl_results, ex02_results);

    dim2 dimGrid = opencl_gridsize(ex02_N);

    clock_t t = ex02_benchmark_start();
    opencl_kernel(ex02_sum_kernel[0], dimGrid, 4,
                  &ex02_cl_gen_randoms.org, sizeof(cl_mem),
                  &ex02_cl_results.org, sizeof(cl_mem)
    );
    ex02_benchmark_stop(t, "sum_gpu");

    opencl_pull_array(ex02_cl_results, ex02_results);
    sum = ex02_results[0];

    printf(" with result %.2f.\n", sum);

    return sum;
}

void ex02(int n, int t, int gpui) {
    ex02_N = n;

    ex02_sum_kernel_init(gpui);

    printf("Calculate SUM of %i randoms:\n", ex02_N);

    int i;
    for(i = 0; i < t; ++i)
    {
        printf("Compare %i:\n", i+1);
        ex02_sum_cpu();
        ex02_sum_gpu(gpui);
    }

    ex02_sum_kernel_deinit(gpui);
}
