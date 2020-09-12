#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "opencl.h"
#include "ex04.h"

int ex04_N = 1;

cl_program* ex04_mul_kernel_program;
cl_kernel* ex04_mul_kernel;

float* ex04_gen_randoms;
cl_mem_ext ex04_cl_gen_randoms;
float* ex04_results;
cl_mem_ext ex04_cl_results;

clock_t ex04_benchmark_start()
{
    clock_t t;
    t = clock();
    return t;
}

void ex04_benchmark_stop(clock_t t, const char* func_name)
{
    t = clock() - t;
    double time_taken = ((double)t);
    printf("Exec of %s took %d ticks", func_name, (int)time_taken);
}

float* ex04_gen_rand(size_t n)
{
    float* set = calloc(n, sizeof(float));
    int i = 0;
    float v = 0;
    for(i = 0; i < n; ++i){
        v = (float)((rand() % 2) + 1);;
        set[i] = v;
    }
    return set;
}

static const char* const ex04_mul_kernel_source = CONVERT_KERNEL_TO_STRING(

        inline void mul(__global float* a, float v)
        {
            float s = 1.f * v;
            float n = 0.f;
            float o = 1.f;
            do {
                n = s * atom_xchg(a, o);
                s = o * atom_xchg(a, n);
            }
            while (s != o);
        }

        __kernel void mul_kernel(__global float *set, __global float* out)
        {
            int i = (get_group_id(0) + get_group_id(1)*get_num_groups(0)) * get_local_size(0) + get_local_id(0);

            mul(out, set[i]);
        }

);

void ex04_mul_kernel_init(int gpui) {
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
            ex04_mul_kernel_program = calloc(ngpu, sizeof(cl_program));
            ex04_mul_kernel = calloc(ngpu, sizeof(cl_kernel));
        }
        opencl_load_buffer(ex04_mul_kernel_source, strlen(ex04_mul_kernel_source),
                           &ex04_mul_kernel_program[d], ngpu, d);
        opencl_create_kernel(&ex04_mul_kernel_program[d], "mul_kernel",
                             &ex04_mul_kernel[d]);
    }
    // Init Mem
    ex04_gen_randoms = ex04_gen_rand(ex04_N);
    ex04_cl_gen_randoms = opencl_make_array(ex04_gen_randoms, sizeof(float), ex04_N);
    ex04_results = calloc(1, sizeof(float));
    ex04_cl_results = opencl_make_array(ex04_results, sizeof(float), 1);
}

void ex04_mul_kernel_deinit(int gpui) {
    int* gpus = calloc(1, sizeof(int));
    gpus[0] = gpui;
    int ngpu = 1;
    // DeInit Mem
    opencl_free(ex04_cl_results);
    opencl_free(ex04_cl_gen_randoms);
    // DeInit GPU Computing
    int d;
    for (d = 0; d < ngpu; ++d) {
        clReleaseKernel(ex04_mul_kernel[d]);
        ex04_mul_kernel[d] = 0;
        clReleaseProgram(ex04_mul_kernel_program[d]);
        ex04_mul_kernel_program[d] = 0;
        if (d == ngpu - 1) {
            free(ex04_mul_kernel_program);
            free(ex04_mul_kernel);
        }
    }
    // DeInint OpenCL
    opencl_deinit(gpus, ngpu);
    free(gpus);
}

float ex04_mul_cpu()
{
    float mul = 1;

    const float* set = ex04_gen_randoms;

    clock_t t = ex04_benchmark_start();
    int i = 0;
    for(i = 0; i < ex04_N; ++i) {
        mul *= set[i];
    }
    ex04_benchmark_stop(t, "mul_cpu");
    printf(" with result %.2f.\n", mul);

    return mul;
}

float ex04_mul_gpu(int gpui)
{
    float mul = 1;

    ex04_results[0] = mul;
    opencl_push_array(ex04_cl_gen_randoms, ex04_gen_randoms);
    opencl_push_array(ex04_cl_results, ex04_results);

    dim2 dimGrid = opencl_gridsize(ex04_N);

    clock_t t = ex04_benchmark_start();
    opencl_kernel(ex04_mul_kernel[0], dimGrid, 4,
                  &ex04_cl_gen_randoms.org, sizeof(cl_mem),
                  &ex04_cl_results.org, sizeof(cl_mem)
    );
    ex04_benchmark_stop(t, "mul_gpu");

    opencl_pull_array(ex04_cl_results, ex04_results);
    mul = ex04_results[0];

    printf(" with result %.2f.\n", mul);

    return mul;
}

void ex04(int n, int t, int gpui) {
    ex04_N = n;

    ex04_mul_kernel_init(gpui);

    printf("Calculate MUL of %i randoms:\n", ex04_N);

    int i;
    for(i = 0; i < t; ++i)
    {
        printf("Compare %i:\n", i+1);
        ex04_mul_cpu();
        ex04_mul_gpu(gpui);
    }

    ex04_mul_kernel_deinit(gpui);
}
