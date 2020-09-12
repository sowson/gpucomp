#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "opencl.h"
#include "ex06.h"

float activation_linear(float x){return x;}
float activation_tanh(float x){return (2.f/(1 + expf(-2.f*x)) - 1);}

int ex06_N = 1;

network* ex06_net;

cl_program* ex06_net_kernel_program;
cl_kernel* ex06_net_kernel;

clock_t ex06_benchmark_start()
{
    clock_t t;
    t = clock();
    return t;
}

void ex06_benchmark_stop(clock_t t, const char* func_name)
{
    t = clock() - t;
    double time_taken = ((double)t);
    printf("Exec of %s took %d ticks", func_name, (int)time_taken);
}

network* ex06_gen_rand(int n, int l)
{
    int i;
    network* net = calloc(1, sizeof(network));
    net->l = l;
    net->L = calloc(l, sizeof(layer));
    for(i = 0; i < l; ++i)
    {
        layer L = net->L[i];
        layer B = net->L[i > 0 ? i-1 : 0];
        L.n = (rand() % n) + n;
        L.b = i == 0 ? 0 : B.n;
        L.w = i == 0 ? 0 : L.n * L.b;
        L.N = calloc(L.n, sizeof(float));
        L.W = i == 0 ? 0 : calloc(L.w, sizeof(float));
        L.a = i < l-1 ? TANH : LINEAR;
        L.A = i < l-1 ? activation_tanh : activation_linear;

        int j;
        for(j = 0; j < L.w; ++j)
        {
            L.W[j] = .01f * (float)(rand() % 4 - 4);
        }

        L.Ng = opencl_make_array(L.N, sizeof(float), L.n);

        if (i > 0)
        {
            L.Wg = opencl_make_array(L.W, sizeof(float), L.w);
            opencl_push_array(L.Wg, L.W);
        }

        net->L[i] = L;
    }
    return net;
}

void ex06_free(network* net)
{
    int i;
    for(i = net->l-1; i >= 0; --i)
    {
        layer L = net->L[i];
        if (i > 0)
        {
            opencl_free(L.Wg);
            if (!L.W) free(L.W);
        }
        opencl_free(L.Ng);
        if (!L.W) free(L.N);
        L.w = 0;
        L.b = 0;
        L.n = 0;
    }
    free(net->L);
    free(net);
}

static const char* const ex06_net_kernel_source = CONVERT_KERNEL_TO_STRING(

        typedef enum {
            LINEAR,
            TANH
        } activation;

        float activation_linear(float x);
        float activation_tanh(float x);

        float activation_linear(float x){return x;}
        float activation_tanh(float x){return (2.f/(1 + exp(-2.f*x)) - 1);}

        __kernel void net_kernel(int b, __global float *B, __global float* W, __global float* N, int a)
        {
            int j = (get_group_id(0) + get_group_id(1)*get_num_groups(0)) * get_local_size(0) + get_local_id(0);

            int w = 0;
            int k = 0;

            N[j] = 0.f;

            for(k = 0; k < b; ++k, ++w) {
                N[j] += B[k] * W[w];
            }

            N[j] = a == LINEAR   ? activation_linear(N[j])  :
                   a == TANH     ? activation_tanh(N[j])    :
                   0;
        }
);

void ex06_net_kernel_init(int gpui) {
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
            ex06_net_kernel_program = calloc(ngpu, sizeof(cl_program));
            ex06_net_kernel = calloc(ngpu, sizeof(cl_kernel));
        }
        opencl_load_buffer(ex06_net_kernel_source, strlen(ex06_net_kernel_source),
                           &ex06_net_kernel_program[d], ngpu, d);
        opencl_create_kernel(&ex06_net_kernel_program[d], "net_kernel",
                             &ex06_net_kernel[d]);
    }
}

void ex06_net_kernel_deinit(int gpui) {
    int* gpus = calloc(1, sizeof(int));
    gpus[0] = gpui;
    int ngpu = 1;
    // DeInit GPU Computing
    int d;
    for (d = 0; d < ngpu; ++d) {
        clReleaseKernel(ex06_net_kernel[d]);
        ex06_net_kernel[d] = 0;
        clReleaseProgram(ex06_net_kernel_program[d]);
        ex06_net_kernel_program[d] = 0;
        if (d == ngpu - 1) {
            free(ex06_net_kernel_program);
            free(ex06_net_kernel);
        }
    }
    // DeInint OpenCL
    opencl_deinit(gpus, ngpu);
    free(gpus);
}

void ex06_net_cpu() {
    network net = *ex06_net;
    int i;
    layer I = net.L[0];
    for(i = 0; i < I.n; ++i)
    {
        I.N[i] = (float)(rand() % 4 + 1);
    }
    clock_t t = ex06_benchmark_start();
    for(i = 1; i < net.l; ++i)
    {
        layer L = net.L[i];
        layer B = net.L[i-1];
        int j;
        int w = 0;
        for(j = 0; j < L.n; ++j) {
            float sum = .0f;
            int k;
            for(k = 0; k < L.b; ++k, ++w) {
                sum += B.N[k] * L.W[w];
            }
            L.N[j] = L.A(sum);
        }
    }
    layer O = net.L[net.l - 1];
    ex06_benchmark_stop(t, "net_cpu");
    printf(" with result %.4f.\n", O.N[0]);
}

void ex06_net_gpu()
{
    network net = *ex06_net;
    int i;
    layer I = net.L[0];
    for(i = 0; i < I.n; ++i)
    {
        I.N[i] = (float)(rand() % 4 + 1);
    }
    opencl_push_array(I.Ng, I.N);
    clock_t t = ex06_benchmark_start();
    for(i = 1; i < net.l; ++i)
    {
        layer L = net.L[i];
        layer B = net.L[i-1];
        dim2 dimGrid = opencl_gridsize(L.n);
        opencl_kernel(ex06_net_kernel[0], dimGrid, 10,
                      &L.b, sizeof(cl_int),
                      &B.Ng.org, sizeof(cl_mem),
                      &L.Wg.org, sizeof(cl_mem),
                      &L.Ng.org, sizeof(cl_mem),
                      &L.a, sizeof(cl_int)
        );
    }
    layer O = net.L[net.l - 1];
    ex06_benchmark_stop(t, "net_gpu");
    opencl_pull_array(O.Ng, O.N);
    printf(" with result %.4f.\n", O.N[0]);
}

void ex06(int n, int l, int t, int gpui) {
    ex06_N = n;

    ex06_net_kernel_init(gpui);
    ex06_net = ex06_gen_rand(ex06_N, l);

    int i;
    for(i = 0; i < t; ++i) {
        printf("Compare %i:\n", i + 1);
        ex06_net_cpu();
        ex06_net_gpu();
    }

    ex06_free(ex06_net);
    ex06_net_kernel_deinit(gpui);
}
