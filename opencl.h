#ifndef OPENCL_H
#define OPENCL_H

//#define GPU_STATS

#ifdef ARM
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#endif

#ifdef __APPLE__
#define CL_SILENCE_DEPRECATION
#include <OpenCL/cl.h>
#include <OpenCL/cl_ext.h>
#else
#include <CL/cl.h>
#include <CL/cl_ext.h>
#endif

#include <math.h>
#include <string.h>

#ifndef MIN
#define MIN(a,b) (((a)<(b))?(a):(b))
#endif
#ifndef MAX
#define MAX(a,b) (((a)>(b))?(a):(b))
#endif

extern int *gpusg;
extern int ngpusg;
extern __thread int opencl_device_id_t;
extern __thread int opencl_device_ct_t;

cl_int *cl_native_double_width_s;
size_t *cl_native_max_group_size_s;
size_t *cl_native_address_bits_s;

typedef struct _cl_mem_ext cl_mem_ext;

typedef struct _cl_mem_ext {
    cl_mem mem;
    cl_mem org;
    size_t len;
    size_t off;
    size_t obs;
    size_t cnt;
    void* ptr;
    void* map;
    cl_command_queue que;
} cl_mem_ext;

cl_context opencl_context;
cl_command_queue* opencl_queues;
cl_device_id* opencl_devices;

extern void test_kernel_gpu(int N, cl_mem_ext input, cl_mem_ext output, cl_mem_ext expected);

typedef struct dim2_
{
    size_t x;
    size_t y;
} dim2;
dim2 dim2_create(const int x, const int y);

#define CONVERT_KERNEL_TO_STRING(...) #__VA_ARGS__

const char* clCheckError(int errorCode);

void opencl_load(const char *fileName, cl_program *output);
void opencl_load_buffer(const char *bufferName, const size_t size, cl_program *output);
void opencl_create_kernel(cl_program *program, const char *kernalName, cl_kernel *kernel);
void opencl_init(int *gpus, int ngpus);
void opencl_deinit(int *gpus, int ngpus);
void opencl_kernel(cl_kernel kernel, const dim2 globalItemSize, const int argc, ...);
void opencl_kernel_local(cl_kernel kernel, const dim2 globalItemSize, const dim2 localItemSize, const int argc, ...);
cl_mem_ext opencl_make_array(void *x, size_t o, size_t n);
void opencl_push_array(cl_mem_ext x_gpu, void *x);
void opencl_pull_array(cl_mem_ext x_gpu, void *x);
void opencl_free(cl_mem_ext x_gpu);
dim2 opencl_gridsize(const int n);
void opencl_dump_mem_stat();

#endif // OPENCL_H
