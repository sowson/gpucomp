#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "opencl.h"
#include "ex05.h"

int ex05_N = 1;

cl_program* ex05_div_kernel_program;
cl_kernel* ex05_div_kernel;

float* ex05_gen_randoms;
cl_mem_ext ex05_cl_gen_randoms;
float* ex05_results;
cl_mem_ext ex05_cl_results;

clock_t ex05_benchmark_start()
{
    clock_t t;
    t = clock();
    return t;
}

void ex05_benchmark_stop(clock_t t, const char* func_name)
{
    t = clock() - t;
    double time_taken = ((double)t);
    printf("Exec of %s took %d ticks", func_name, (int)time_taken);
}

float* ex05_gen_rand(size_t n)
{
    float* set = calloc(n, sizeof(float));
    int i = 0;
    float v = 0;
    for(i = 0; i < n; ++i){
        v = (float)((rand() % 4) + 1);
        set[i] = v;
    }
    return set;
}

static const char* const ex05_div_kernel_source = CONVERT_KERNEL_TO_STRING(

        inline void div(__global float* a, float v)
        {
            float s = 1.f / v;
            float n = 0.f;
            float o = 1.f;
            do {
                n = s * atom_xchg(a, o);
                s = o * atom_xchg(a, n);
            }
            while (s != o);
        }

        __kernel void div_kernel(__global float *set, __global float* out)
        {
            int i = (get_group_id(0) + get_group_id(1)*get_num_groups(0)) * get_local_size(0) + get_local_id(0);

            div(out, set[i]);
        }

);

void ex05_div_kernel_init(int gpui) {
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
            ex05_div_kernel_program = calloc(ngpu, sizeof(cl_program));
            ex05_div_kernel = calloc(ngpu, sizeof(cl_kernel));
        }
        opencl_load_buffer(ex05_div_kernel_source, strlen(ex05_div_kernel_source),
                           &ex05_div_kernel_program[d]);
        opencl_create_kernel(&ex05_div_kernel_program[d], "div_kernel",
                             &ex05_div_kernel[d]);
    }
    // Init Mem
    ex05_gen_randoms = ex05_gen_rand(ex05_N);
    ex05_cl_gen_randoms = opencl_make_array(ex05_gen_randoms, sizeof(float), ex05_N);
    ex05_results = calloc(1, sizeof(float));
    ex05_cl_results = opencl_make_array(ex05_results, sizeof(float), 1);
}

void ex05_div_kernel_deinit(int gpui) {
    // DeInint OpenCL
    int ngpu = 1;
    int* gpus = calloc(ngpu, sizeof(int));
    gpus[0] = gpui;
    // DeInit Mem
    opencl_free(ex05_cl_results);
    opencl_free(ex05_cl_gen_randoms);
    // DeInit GPU Computing
    int d;
    for (d = 0; d < ngpu; ++d) {
        clReleaseKernel(ex05_div_kernel[d]);
        ex05_div_kernel[d] = 0;
        clReleaseProgram(ex05_div_kernel_program[d]);
        ex05_div_kernel_program[d] = 0;
        if (d == ngpu - 1) {
            free(ex05_div_kernel_program);
            free(ex05_div_kernel);
        }
    }
    // DeInint OpenCL
    opencl_deinit(gpus, ngpu);
    free(gpus);
}

float ex05_div_cpu()
{
    float div = (float)ex05_N*1000000000000000000;

    const float* set = ex05_gen_randoms;

    clock_t t = ex05_benchmark_start();
    int i = 0;
    for(i = 0; i < ex05_N; ++i) {
        div /= set[i];
    }
    ex05_benchmark_stop(t, "div_cpu");
    printf(" with result %.4f.\n", div);

    return div;
}

float ex05_div_gpu(int gpui)
{
    float div = (float)ex05_N*1000000000000000000;

    ex05_results[0] = div;
    opencl_push_array(ex05_cl_gen_randoms, ex05_gen_randoms);
    opencl_push_array(ex05_cl_results, ex05_results);

    dim2 dimGrid = opencl_gridsize(ex05_N);

    clock_t t = ex05_benchmark_start();
    opencl_kernel(ex05_div_kernel[0], dimGrid, 4,
                  &ex05_cl_gen_randoms.org, sizeof(cl_mem),
                  &ex05_cl_results.org, sizeof(cl_mem)
    );
    ex05_benchmark_stop(t, "div_gpu");

    opencl_pull_array(ex05_cl_results, ex05_results);
    div = ex05_results[0];

    printf(" with result %.4f.\n", div);

    return div;
}

void ex05(int n, int t, int gpui) {
    ex05_N = n;

    ex05_div_kernel_init(gpui);

    printf("Calculate DIV of %i randoms:\n", ex05_N);

    int i;
    for(i = 0; i < t; ++i)
    {
        printf("Compare %i:\n", i+1);
        ex05_div_cpu();
        ex05_div_gpu(gpui);
    }

    ex05_div_kernel_deinit(gpui);
}
