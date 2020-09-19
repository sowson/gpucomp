#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "opencl.h"
#include "ex03.h"

int ex03_N = 1;

cl_program* ex03_sub_kernel_program;
cl_kernel* ex03_sub_kernel;

float* ex03_gen_randoms;
cl_mem_ext ex03_cl_gen_randoms;
float* ex03_results;
cl_mem_ext ex03_cl_results;

clock_t ex03_benchmark_start()
{
    clock_t t;
    t = clock();
    return t;
}

void ex03_benchmark_stop(clock_t t, const char* func_name)
{
    t = clock() - t;
    double time_taken = ((double)t);
    printf("Exec of %s took %d ticks", func_name, (int)time_taken);
}

float* ex03_gen_rand(size_t n)
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

static const char* const ex03_sub_kernel_source = CONVERT_KERNEL_TO_STRING(

        inline void sub(__global float* a, float v)
        {
            float s = -1.f * v;
            float n = 0.f;
            float o = 0.f;
            do {
                n = s + atom_xchg(a, o);
                s = o + atom_xchg(a, n);
            }
            while (s != o);
        }

        __kernel void sub_kernel(__global float *set, __global float* out)
        {
            int i = (get_group_id(0) + get_group_id(1)*get_num_groups(0)) * get_local_size(0) + get_local_id(0);

            sub(out, set[i]);
        }

);

void ex03_sub_kernel_init(int gpui) {
    // Inint OpenCL
    int ngpu = 1;
    int* gpus = calloc(ngpu, sizeof(int));
    gpus[0] = gpui;
    opencl_init(gpus, ngpu);
    free(gpus);
    // Init GPU Computing
    int d;
    for (d = 0; d < ngpu; ++d) {
        if (d == 0) {
            ex03_sub_kernel_program = calloc(ngpu, sizeof(cl_program));
            ex03_sub_kernel = calloc(ngpu, sizeof(cl_kernel));
        }
        opencl_load_buffer(ex03_sub_kernel_source, strlen(ex03_sub_kernel_source),
                           &ex03_sub_kernel_program[d]);
        opencl_create_kernel(&ex03_sub_kernel_program[d], "sub_kernel",
                             &ex03_sub_kernel[d]);
    }
    // Init Mem
    ex03_gen_randoms = ex03_gen_rand(ex03_N);
    ex03_cl_gen_randoms = opencl_make_array(ex03_gen_randoms, sizeof(float), ex03_N);
    ex03_results = calloc(1, sizeof(float));
    ex03_cl_results = opencl_make_array(ex03_results, sizeof(float), 1);
}

void ex03_sub_kernel_deinit(int gpui) {
    // DeInint OpenCL
    int ngpu = 1;
    int* gpus = calloc(ngpu, sizeof(int));
    gpus[0] = gpui;
    // DeInit Mem
    opencl_free(ex03_cl_results);
    opencl_free(ex03_cl_gen_randoms);
    // DeInit GPU Computing
    int d;
    for (d = 0; d < ngpu; ++d) {
        clReleaseKernel(ex03_sub_kernel[d]);
        ex03_sub_kernel[d] = 0;
        clReleaseProgram(ex03_sub_kernel_program[d]);
        ex03_sub_kernel_program[d] = 0;
        if (d == ngpu - 1) {
            free(ex03_sub_kernel_program);
            free(ex03_sub_kernel);
        }
    }
    // DeInint OpenCL
    opencl_deinit(gpus, ngpu);
    free(gpus);
}

float ex03_sub_cpu()
{
    float sub = ex03_N*1000;

    const float* set = ex03_gen_randoms;

    clock_t t = ex03_benchmark_start();
    int i = 0;
    for(i = 0; i < ex03_N; ++i) {
        sub -= set[i];
    }
    ex03_benchmark_stop(t, "sub_cpu");
    printf(" with result %.2f.\n", sub);

    return sub;
}

float ex03_sub_gpu(int gpui)
{
    float sub = ex03_N*1000;

    ex03_results[0] = sub;
    opencl_push_array(ex03_cl_gen_randoms, ex03_gen_randoms);
    opencl_push_array(ex03_cl_results, ex03_results);

    dim2 dimGrid = opencl_gridsize(ex03_N);

    clock_t t = ex03_benchmark_start();
    opencl_kernel(ex03_sub_kernel[0], dimGrid, 4,
                  &ex03_cl_gen_randoms.org, sizeof(cl_mem),
                  &ex03_cl_results.org, sizeof(cl_mem)
    );
    ex03_benchmark_stop(t, "sub_gpu");

    opencl_pull_array(ex03_cl_results, ex03_results);
    sub = ex03_results[0];

    printf(" with result %.2f.\n", sub);

    return sub;
}

void ex03(int n, int t, int gpui) {
    ex03_N = n;

    ex03_sub_kernel_init(gpui);

    printf("Calculate SUB of %i randoms:\n", ex03_N);

    int i;
    for(i = 0; i < t; ++i)
    {
        printf("Compare %i:\n", i+1);
        ex03_sub_cpu();
        ex03_sub_gpu(gpui);
    }

    ex03_sub_kernel_deinit(gpui);
}
