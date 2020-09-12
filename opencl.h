#ifndef OPENCL_H
#define OPENCL_H

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#ifdef __APPLE__
#define CL_SILENCE_DEPRECATION
#include <OpenCL/cl.h>
#include <OpenCL/cl_ext.h>
#else
#include <CL/cl.h>
#include <CL/cl_ext.h>
#endif

#include <string.h>

extern cl_int *cl_native_double_width_s;
extern size_t *cl_native_max_group_size_s;
extern size_t *cl_native_address_bits_s;

typedef struct _cl_mem_ext cl_mem_ext;

typedef struct _cl_mem_ext {
    cl_mem mem;
    cl_mem org;
    size_t len;
    size_t obs;
    void* ptr;
    void* map;
    cl_command_queue que;
} cl_mem_ext;

cl_context opencl_context;
cl_command_queue opencl_queue;
cl_device_id* opencl_devices;

typedef struct dim2_
{
    size_t x;
    size_t y;
} dim2;
dim2 dim2_create(int x, int y);

#define CONVERT_KERNEL_TO_STRING(...) #__VA_ARGS__

dim2 opencl_gridsize(int n);

void opencl_init(int *gpus, int ngpus);
void opencl_deinit(int *gpus, int ngpus);

void opencl_load(char *fileName, cl_program *output, int ngpus, int gpui);
void opencl_load_buffer(const char *buffer, size_t size, cl_program *output, int ngpus, int gpui);
void opencl_create_kernel(cl_program *program, char *kernelName, cl_kernel *kernel);
void opencl_kernel(cl_kernel kernel, dim2 globalItemSize, int argc, ...);

cl_mem_ext opencl_make_array(void *x, size_t o, size_t n);

void opencl_push_array(cl_mem_ext x_gpu, void *x);
void opencl_pull_array(cl_mem_ext x_gpu, void *x);

void opencl_free(cl_mem_ext x_gpu);

char* clGetErrorString(int errorCode);
char* clCheckError(int errorCode);

#endif // OPENCL_H