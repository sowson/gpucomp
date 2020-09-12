#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "opencl.h"

typedef enum {
    LINEAR,
    TANH
} activation;

float activation_linear(float x);
float activation_tanh(float x);

typedef struct _layer layer;
typedef struct _layer {
    int n; // <= neurons here
    int w; // <= n * b compute
    int b; // <= neurons before
    float* N;
    cl_mem_ext Ng;
    float* W;
    cl_mem_ext Wg;
    activation a;
    float (*A) (float x);
} layer;

typedef struct _network network;
typedef struct _network {
    int l; // <= layers count
    layer* L;
} network;

int ex06_N ;
network* ex06_net;;
cl_program* ex06_net_kernel_program;
cl_kernel* ex06_net_kernel;
clock_t ex06_benchmark_start();
void ex06_benchmark_stop(clock_t t, const char* func_name);
network* ex06_gen_rand(int n, int l);
void ex06_free(network* net);
static const char* const ex06_net_kernel_source;
void ex06_net_kernel_init(int gpui);
void ex06_net_kernel_deinit(int gpui);
void ex06_net_cpu();
void ex06_net_gpu();
void ex06(int n, int l, int t, int gpui);