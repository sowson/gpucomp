#include <unistd.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <math.h>
#include <stdio.h>
#include "opencl.h"

cl_int *cl_native_double_width_s;
size_t *cl_native_max_group_size_s;
size_t *cl_native_address_bits_s;

cl_context_properties* cl_props;

char* concat(char *s1, char *s2)
{
    char *result = calloc(strlen(s1) + strlen(s2) + 1, sizeof(char));
    strcpy(result, s1);
    strcat(result, s2);
    return result;
}

dim2 dim2_create(int x, int y)
{
    dim2 ret;

    ret.x = x;
    ret.y = y;

    return ret;
}

dim2 opencl_gridsize(int n)
{
    dim2 ret = dim2_create(n, 1);

    return ret;
}

void opencl_init(int *gpus, int ngpus)
{
    cl_native_double_width_s = calloc(ngpus, sizeof(int));
    cl_native_max_group_size_s = calloc(ngpus, sizeof(int));
    cl_native_address_bits_s = calloc(ngpus, sizeof(int));

    cl_int clErr;

    cl_platform_id clPlatform = 0;
    cl_uint clNumPlatforms = 0;

    cl_props = calloc(3, sizeof(cl_context_properties));
    cl_props[0] = CL_CONTEXT_PLATFORM;
    cl_props[1] = 0;
    cl_props[2] = 0;

    clErr = clGetPlatformIDs(CL_TRUE, &clPlatform, &clNumPlatforms);

    if (clErr != CL_SUCCESS) {
        printf("CL: opencl_init: Could not get platform IDs.\n");
        return;
    }

    cl_uint num = 32;
    cl_uint all = 0;
    cl_device_id devices[num];
    clErr = clGetDeviceIDs(clPlatform, CL_DEVICE_TYPE_ALL, num, devices, &all);

    if (clErr != CL_SUCCESS) {
        printf("CL: opencl_init: Could not get device IDs.\n");
        return;
    }

    opencl_devices = (cl_device_id *) calloc((cl_uint)ngpus, sizeof(cl_device_id));

    int i;
    for(i = 0; i < ngpus; ++i)
    {
        opencl_devices[i] = devices[gpus[i]];
    }

    cl_props[1] = (cl_context_properties) clPlatform;

    opencl_context = clCreateContext(cl_props, (cl_uint)ngpus,
                                     opencl_devices, NULL, NULL, &clErr);

    if (clErr != CL_SUCCESS) {
        printf("CL: opencl_init: Could not create context.\n");
        return;
    }

    opencl_queue = clCreateCommandQueue(opencl_context, *opencl_devices, CL_FALSE, &clErr);
    if (clErr != CL_SUCCESS) {
        printf("CL: opencl_init: Could not create queue.\n");
        return;
    }

    printf("OpenCL Platform\n");
    int d;
    for (d = 0; d < ngpus; ++d) {
        cl_native_double_width_s[d] = 0;
        cl_native_max_group_size_s[d] = 0;
        cl_native_address_bits_s[d] = 0;

        clGetDeviceInfo(opencl_devices[d], CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE, sizeof(cl_uint), &cl_native_double_width_s[d], NULL);
        clGetDeviceInfo(opencl_devices[d], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &cl_native_max_group_size_s[d], NULL);
        clGetDeviceInfo(opencl_devices[d], CL_DEVICE_ADDRESS_BITS, sizeof(size_t), &cl_native_address_bits_s[d], NULL);

        size_t bufferSize = 2048;
        char *buffer = (char *) calloc(bufferSize, sizeof(char));

        printf("Device ID: %d\n", gpus[d]);
        clGetDeviceInfo(opencl_devices[d], CL_DEVICE_NAME, bufferSize * sizeof(char), buffer, NULL);
        printf("Device Name: %s\n", buffer);
        clGetDeviceInfo(opencl_devices[d], CL_DEVICE_VENDOR, bufferSize * sizeof(char), buffer, NULL);
        printf("Device Vendor: %s\n", buffer);
        clGetDeviceInfo(opencl_devices[d], CL_DEVICE_VERSION, bufferSize * sizeof(char), buffer, NULL);
        printf("Device OpenCL Availability: %s\n", buffer);
        clGetDeviceInfo(opencl_devices[d], CL_DRIVER_VERSION, bufferSize * sizeof(char), buffer, NULL);
        printf("Device OpenCL Used: %s\n", buffer);
        printf("Device Double Precision: %s\n", cl_native_double_width_s[d] == 0 ? "NO" : "YES");
        printf("Device Max Group Size: %zu\n", cl_native_max_group_size_s[d]);
        printf("Device Address Bits: %zu\n", cl_native_address_bits_s[d]);
        free(buffer);
    }
    printf("\n");
}

void opencl_deinit(int *gpus, int ngpus)
{
    int a;
    int d;
    for (a = 0, d = ngpus-1; a < ngpus; --d, ++a) {
        d = a;

        clFinish(opencl_queue);
        clReleaseCommandQueue(opencl_queue);
    }

    free(cl_props);
    clReleaseContext(opencl_context);
    opencl_context = 0;

    free(opencl_devices);

    free(cl_native_double_width_s);
    free(cl_native_max_group_size_s);
    free(cl_native_address_bits_s);
}

void opencl_load(char *fileName, cl_program *output, int ngpus, int gpui)
{
    FILE *fp;
    size_t lSize, readSize;
    char * sourceBuffer;

    fp = fopen(fileName, "r");

    if (fp == NULL)
    {
        printf("CL: opencl_load: Could not open file: %s\n", fileName);
        fclose(fp);
        return;
    }

    // Determine file size.
    fseek(fp, 0, SEEK_END);
    lSize = ftell(fp);
    rewind(fp);

    sourceBuffer = (char*) malloc(sizeof(char) * lSize);

    if (sourceBuffer == NULL)
    {
        printf("CL: opencl_load: Could not allocate memory for file: %s\n",
               fileName);
        fclose(fp);
        return;
    }

    readSize = fread(sourceBuffer, 1, lSize, fp);
    fclose(fp);

    if (readSize > lSize)
    {
        printf("CL: opencl_load: failed to read file: %s\n", fileName);
        free(sourceBuffer);
        return;
    }

    opencl_load_buffer(sourceBuffer, readSize, output, ngpus, gpui);

    free(sourceBuffer);
}

void opencl_load_buffer(const char *buffer, size_t size, cl_program *output, int ngpus, int gpui)
{
    cl_int clErr;

    *output = clCreateProgramWithSource(opencl_context, CL_TRUE,
                                        (const char**)&buffer, &size, &clErr);

    if (clErr != CL_SUCCESS)
    {
        printf("opencl_load: could not create program. error: %s\n", clCheckError(clErr));
        return;
    }

    clErr = clBuildProgram(
            *output,
            1,
            opencl_devices,
            NULL, NULL, NULL);

    if (clErr != CL_SUCCESS)
    {
        printf("opencl_load: could not compile. error: %s\n", clCheckError(clErr));
        size_t len;
        char *ebuffer = (char*)calloc(0x10000000, sizeof(char));
        clGetProgramBuildInfo(*output, opencl_devices[0], CL_PROGRAM_BUILD_LOG, 0x10000000 * sizeof(char), ebuffer, &len);
        printf("CL_PROGRAM_BUILD_LOG:\n%s\n", ebuffer);
        printf("CODE:\n%s\n", buffer);
        free(ebuffer);
        exit(1);
    }
}

void opencl_create_kernel(cl_program *program, char *kernelName, cl_kernel *kernel)
{
    cl_int clErr;

    *kernel = clCreateKernel(*program, kernelName, &clErr);

    if (clErr)
    {
        printf("CL: opencl_create_kernel: Could not create kernel %s.\n",
               kernelName);
    }
}

void opencl_kernel(cl_kernel kernel, dim2 globalItemSize, int argc, ...)
{
    cl_int clErr;

    va_list vl;
    va_start(vl, argc);

    size_t argSize = 0;
    void *argValue = NULL;

    int i, j;
    for (i = 0, j = 0; i < argc; i+=2, ++j)
    {
        argValue = va_arg(vl, void*);
        argSize = va_arg(vl, size_t);

        assert(argValue);

        clErr = clSetKernelArg(kernel, j, argSize, argValue);

        if (clErr != CL_SUCCESS)
        {
            size_t bufferSize = 2048;
            char *kernelName = (char*) calloc(bufferSize, sizeof(char));

            clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, bufferSize, kernelName, NULL);
            printf("CL: opencl_kernel %s could not set kernel argument. error: %s\n", kernelName, clCheckError(clErr));

            free(kernelName);
        }
    }

    va_end(vl);

    size_t globalOffser[2], globalItems[2];
    globalOffser[0] = 0;
    globalOffser[1] = 0;
    globalItems[0] = globalItemSize.x;
    globalItems[1] = globalItemSize.y;

    clErr = clEnqueueNDRangeKernel(opencl_queue, kernel, 2,
                                   globalOffser, globalItems, NULL, 0, NULL, NULL);

    if (clErr != CL_SUCCESS)
    {
        size_t bufferSize = 2048;
        char *kernelName = (char*) calloc(bufferSize, sizeof(char));

        clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, bufferSize, kernelName, NULL);
        printf("CL: opencl %s error: %s\n", kernelName, clCheckError(clErr));

        free(kernelName);
    }
}

cl_mem_ext opencl_make_array(void *x, size_t o, size_t n)
{
    cl_mem_ext buf;

    buf.len = n;
    buf.obs = o;

    buf.ptr = x;

    cl_int clErr;
    buf.org = clCreateBuffer(opencl_context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR,
                             buf.len * buf.obs, buf.ptr, &clErr);
    if (clErr != CL_SUCCESS)
        printf("CL: could create buffer on device. error: %s\n", clCheckError(clErr));

    buf.mem = buf.org;
    buf.que = opencl_queue;

    return buf;
}

void opencl_push_array(cl_mem_ext x_gpu, void *x)
{
    cl_int clErr;

    if (x_gpu.ptr == (void*)x) {

        clErr = clEnqueueWriteBuffer(x_gpu.que, x_gpu.mem, CL_TRUE, 0, x_gpu.len * x_gpu.obs, x, 0, NULL, NULL);
        if (clErr != CL_SUCCESS)
            printf("CL: could not push array to device. error: %s\n", clCheckError(clErr));

    } else {
        x_gpu.map = clEnqueueMapBuffer(x_gpu.que, x_gpu.org, CL_TRUE, CL_MAP_WRITE,
                                       0, x_gpu.len * x_gpu.obs, 0, NULL, NULL, &clErr);

        if (clErr != CL_SUCCESS)
            printf("CL: could not map array to device. error: %s\n", clCheckError(clErr));

        memcpy(x_gpu.map, x, x_gpu.len * x_gpu.obs);

        clErr = clEnqueueUnmapMemObject(x_gpu.que, x_gpu.org, x_gpu.map, 0, NULL, NULL);

        if (clErr != CL_SUCCESS)
            printf("CL: could not unmap array from device. error: %s\n", clCheckError(clErr));
    }
}

void opencl_pull_array(cl_mem_ext x_gpu, void *x)
{
    cl_int clErr;

    if (x_gpu.ptr == (void*)x) {

        clErr = clEnqueueReadBuffer(x_gpu.que, x_gpu.mem, CL_TRUE, 0, x_gpu.len * x_gpu.obs, x, 0, NULL, NULL);
        if (clErr != CL_SUCCESS)
            printf("CL: could not pull array from device. error: %s\n", clCheckError(clErr));

    } else {
        x_gpu.map = clEnqueueMapBuffer(x_gpu.que, x_gpu.org, CL_TRUE, CL_MAP_READ,
                                       0, x_gpu.len * x_gpu.obs, 0, NULL, NULL, &clErr);

        if (clErr != CL_SUCCESS)
            printf("CL: could not map array to device. error: %s\n", clCheckError(clErr));

        memcpy(x, x_gpu.map, x_gpu.len * x_gpu.obs);

        clErr = clEnqueueUnmapMemObject(x_gpu.que, x_gpu.org, x_gpu.map, 0, NULL, NULL);

        if (clErr != CL_SUCCESS)
            printf("CL: could not unmap array from device. error: %s\n", clCheckError(clErr));
    }
}

void opencl_free(cl_mem_ext x_gpu)
{
    x_gpu.len = 0;
    x_gpu.obs = 0;
    x_gpu.mem = 0;
    clReleaseMemObject(x_gpu.org);
    x_gpu.org = 0;
    x_gpu.map = 0;
    x_gpu.que = 0;
    if(!x_gpu.ptr) free(x_gpu.ptr);
    x_gpu.ptr = 0;
}

void opencl_copy(cl_mem_ext y_gpu, cl_mem_ext x_gpu)
{
    y_gpu.len = x_gpu.len;
    y_gpu.obs = x_gpu.obs;
    y_gpu.mem = x_gpu.mem;
    y_gpu.org = x_gpu.org;
    y_gpu.map = x_gpu.map;
    y_gpu.que = x_gpu.que;
    y_gpu.ptr = x_gpu.ptr;
}

void opencl_dump_mem_stat(int gpui)
{
    size_t used, total;

    clGetDeviceInfo(opencl_devices[gpui], CL_DEVICE_GLOBAL_MEM_SIZE,
                    sizeof(size_t), &total, NULL);

    clGetDeviceInfo(opencl_devices[gpui], CL_DEVICE_LOCAL_MEM_SIZE,
                    sizeof(size_t), &used, NULL);

    printf("OpenCL memory status: Used/Free/Total = [%lu]/[%lu]/[%lu]\n", used, total - used, total);
}

char* clGetErrorString(int errorCode)
{
    switch (errorCode) {
        case 0: return "CL_SUCCESS";
        case -1: return "CL_DEVICE_NOT_FOUND";
        case -2: return "CL_DEVICE_NOT_AVAILABLE";
        case -3: return "CL_COMPILER_NOT_AVAILABLE";
        case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case -5: return "CL_OUT_OF_RESOURCES";
        case -6: return "CL_OUT_OF_HOST_MEMORY";
        case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case -8: return "CL_MEM_COPY_OVERLAP";
        case -9: return "CL_IMAGE_FORMAT_MISMATCH";
        case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case -12: return "CL_MAP_FAILURE";
        case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
        case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
        case -15: return "CL_COMPILE_PROGRAM_FAILURE";
        case -16: return "CL_LINKER_NOT_AVAILABLE";
        case -17: return "CL_LINK_PROGRAM_FAILURE";
        case -18: return "CL_DEVICE_PARTITION_FAILED";
        case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
        case -30: return "CL_INVALID_VALUE";
        case -31: return "CL_INVALID_DEVICE_TYPE";
        case -32: return "CL_INVALID_PLATFORM";
        case -33: return "CL_INVALID_DEVICE";
        case -34: return "CL_INVALID_CONTEXT";
        case -35: return "CL_INVALID_QUEUE_PROPERTIES";
        case -36: return "CL_INVALID_COMMAND_QUEUE";
        case -37: return "CL_INVALID_HOST_PTR";
        case -38: return "CL_INVALID_MEM_OBJECT";
        case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case -40: return "CL_INVALID_IMAGE_SIZE";
        case -41: return "CL_INVALID_SAMPLER";
        case -42: return "CL_INVALID_BINARY";
        case -43: return "CL_INVALID_BUILD_OPTIONS";
        case -44: return "CL_INVALID_PROGRAM";
        case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
        case -46: return "CL_INVALID_KERNEL_NAME";
        case -47: return "CL_INVALID_KERNEL_DEFINITION";
        case -48: return "CL_INVALID_KERNEL";
        case -49: return "CL_INVALID_ARG_INDEX";
        case -50: return "CL_INVALID_ARG_VALUE";
        case -51: return "CL_INVALID_ARG_SIZE";
        case -52: return "CL_INVALID_KERNEL_ARGS";
        case -53: return "CL_INVALID_WORK_DIMENSION";
        case -54: return "CL_INVALID_WORK_GROUP_SIZE";
        case -55: return "CL_INVALID_WORK_ITEM_SIZE";
        case -56: return "CL_INVALID_GLOBAL_OFFSET";
        case -57: return "CL_INVALID_EVENT_WAIT_LIST";
        case -58: return "CL_INVALID_EVENT";
        case -59: return "CL_INVALID_OPERATION";
        case -60: return "CL_INVALID_GL_OBJECT";
        case -61: return "CL_INVALID_BUFFER_SIZE";
        case -62: return "CL_INVALID_MIP_LEVEL";
        case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
        case -64: return "CL_INVALID_PROPERTY";
        case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
        case -66: return "CL_INVALID_COMPILER_OPTIONS";
        case -67: return "CL_INVALID_LINKER_OPTIONS";
        case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";
        case -69: return "CL_INVALID_PIPE_SIZE";
        case -70: return "CL_INVALID_DEVICE_QUEUE";
        case -71: return "CL_INVALID_SPEC_ID";
        case -72: return "CL_MAX_SIZE_RESTRICTION_EXCEEDED";
        case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
        case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
        case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
        case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
        case -1006: return "CL_INVALID_D3D11_DEVICE_KHR";
        case -1007: return "CL_INVALID_D3D11_RESOURCE_KHR";
        case -1008: return "CL_D3D11_RESOURCE_ALREADY_ACQUIRED_KHR";
        case -1009: return "CL_D3D11_RESOURCE_NOT_ACQUIRED_KHR";
        case -1010: return "CL_INVALID_DX9_MEDIA_ADAPTER_KHR";
        case -1011: return "CL_INVALID_DX9_MEDIA_SURFACE_KHR";
        case -1012: return "CL_DX9_MEDIA_SURFACE_ALREADY_ACQUIRED_KHR";
        case -1013: return "CL_DX9_MEDIA_SURFACE_NOT_ACQUIRED_KHR";
        case -1093: return "CL_INVALID_EGL_OBJECT_KHR";
        case -1092: return "CL_EGL_RESOURCE_NOT_ACQUIRED_KHR";
        case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
        case -1057: return "CL_DEVICE_PARTITION_FAILED_EXT";
        case -1058: return "CL_INVALID_PARTITION_COUNT_EXT";
        case -1059: return "CL_INVALID_PARTITION_NAME_EXT";
        case -1094: return "CL_INVALID_ACCELERATOR_INTEL";
        case -1095: return "CL_INVALID_ACCELERATOR_TYPE_INTEL";
        case -1096: return "CL_INVALID_ACCELERATOR_DESCRIPTOR_INTEL";
        case -1097: return "CL_ACCELERATOR_TYPE_NOT_SUPPORTED_INTEL";
        case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
        case -1098: return "CL_INVALID_VA_API_MEDIA_ADAPTER_INTEL";
        case -1099: return "CL_INVALID_VA_API_MEDIA_SURFACE_INTEL";
        case -1100: return "CL_VA_API_MEDIA_SURFACE_ALREADY_ACQUIRED_INTEL";
        case -1101: return "CL_VA_API_MEDIA_SURFACE_NOT_ACQUIRED_INTEL";
        default: return "CL_UNKNOWN_ERROR";
    }
}

char* clCheckError(int errorCode)
{
    return clGetErrorString(errorCode);
}
