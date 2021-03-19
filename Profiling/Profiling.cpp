//------------------------------------------------------------------------------
//
// Name:       vadd.c
// 
// Purpose:    Elementwise addition of two vectors (c = a + b)
//             Use the profiling interface with events to time the vadd
//
// HISTORY:    Written by Tim Mattson, June 2011
//             
//------------------------------------------------------------------------------


#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#ifdef APPLE
#include <OpenCL/opencl.h>
#include <unistd.h>
#else
#include "CL/cl.h"
#endif

//------------------------------------------------------------------------------

#define TOL    (0.001)   // tolerance used in floating point comparisons
#define LENGTH (1024)    // length of vectors a, b, and c

//------------------------------------------------------------------------------
//
// kernel:  vadd  
//
// Purpose: Compute the elementwise sum c = a+b
// 
// input: a and b float vectors of length count
//
// output: c float vector of length count holding the sum a + b
//

const char* KernelSource = "\n" \
"__kernel void vadd(                                                         \n" \
"   __global float* a,                                                  \n" \
"   __global float* b,                                                  \n" \
"   __global float* c,                                                  \n" \
"   const unsigned int count)                                           \n" \
"{                                                                      \n" \
"   int i = get_global_id(0);                                           \n" \
"   if(i < count)                                                       \n" \
"       c[i] = a[i] + b[i];                                             \n" \
"}                                                                      \n" \
"\n";

//------------------------------------------------------------------------------


int main(int argc, char** argv)
{
    int          err;                   // error code returned from OpenCL calls
    float        a_data[LENGTH];        // a vector 
    float        b_data[LENGTH];        // b vector 
    float        c_res[LENGTH];        // c vector (a+b) returned from the compute device
    unsigned int correct;               // number of correct results  

    size_t global;                      // global domain size  
    size_t local;                       // local  domain size  

    cl_device_id     device_id;         // compute device id 
    cl_context       context;           // compute context
    cl_command_queue commands;          // compute command queue
    cl_program       program;           // compute program
    cl_kernel        kernel;            // compute kernel

    cl_mem a_in;                        // device memory used for the input  a vector
    cl_mem b_in;                        // device memory used for the input  b vector
    cl_mem c_out;                       // device memory used for the output c vector

    // Fill vectors a and b with random float values
    int i = 0;
    int count = LENGTH;
    for (i = 0; i < count; i++) {
        a_data[i] = rand() / (float)RAND_MAX;
        b_data[i] = rand() / (float)RAND_MAX;
    }

    // use whichever one is "first"
    cl_uint numPlatforms;
    cl_platform_id firstPlatformId;

    err = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms);
    if (err != CL_SUCCESS || numPlatforms <= 0)
    {
        printf("Error: Failed to find the platform!\n");
        return EXIT_FAILURE;
    }

    err = clGetDeviceIDs(firstPlatformId, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to create a device group!\n");
        return EXIT_FAILURE;
    }

    // Create a compute context 
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if (!context)
    {
        printf("Error: Failed to create a compute context!\n");
        return EXIT_FAILURE;
    }

    // Create a command queue ... enable profiling
    commands = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
    if (!commands)
    {
        printf("Error: Failed to create a command commands!\n");
        return EXIT_FAILURE;
    }

    // Create the compute program from the source buffer
    program = clCreateProgramWithSource(context, 1, (const char**)&KernelSource, NULL, &err);
    if (!program)
    {
        printf("Error: Failed to create compute program!\n");
        return EXIT_FAILURE;
    }

    // Build the program  
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        exit(1);
    }

    // Create the compute kernel from the program 
    kernel = clCreateKernel(program, "vadd", &err);
    if (!kernel || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel!\n");
        exit(1);
    }

    // Create the input (a, b) and output (c) arrays in device memory  
    a_in = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * count, NULL, NULL);
    b_in = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * count, NULL, NULL);
    c_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * count, NULL, NULL);
    if (!a_in || !b_in || !c_out)
    {
        printf("Error: Failed to allocate device memory!\n");
        exit(1);
    }

    // Write a and b vectors into compute device memory 
    err = clEnqueueWriteBuffer(commands, a_in, CL_TRUE, 0, sizeof(float) * count, a_data, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write a_data to source array!\n");
        exit(1);
    }
    err = clEnqueueWriteBuffer(commands, b_in, CL_TRUE, 0, sizeof(float) * count, b_data, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write b_data to source array!\n");
        exit(1);
    }

    // Set the arguments to our compute kernel
    err = 0;
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_in);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_in);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &c_out);
    err |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &count);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        exit(1);
    }

    // Get the maximum work group size for executing the kernel on the device
    err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to retrieve kernel work group info! %d\n", err);
        exit(1);
    }
    double rtime;
    rtime = clock();

    // Execute the kernel over the entire range of our 1d input data set
    // using the maximum number of work group items for this device
    cl_event prof_event;
    global = count;
    err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &local, 0, NULL, &prof_event);
    if (err)
    {
        printf("Error: Failed to execute kernel!\n");
        return EXIT_FAILURE;
    }

    // Wait for the commands to complete before reading back results
    clFinish(commands);
    rtime = clock() - rtime;

    printf("\nThe kernel ran in %lf seconds\n", rtime);

    // extract timing data from the event, prof_event
    err = clWaitForEvents(1, &prof_event);
    cl_ulong ev_start_time = (cl_ulong)0;
    cl_ulong ev_end_time = (cl_ulong)0;
    err = clGetEventProfilingInfo(prof_event, CL_PROFILING_COMMAND_START,
        sizeof(cl_ulong), &ev_start_time, NULL);
    err = clGetEventProfilingInfo(prof_event, CL_PROFILING_COMMAND_END,
        sizeof(cl_ulong), &ev_end_time, NULL);
    printf("prof says %f secs \n", (double)(ev_end_time - ev_start_time) * 1.0e-9);



    // Read back the results from the compute device
    err = clEnqueueReadBuffer(commands, c_out, CL_TRUE, 0, sizeof(float) * count, c_res, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array! %d\n", err);
        exit(1);
    }

    // Test the results
    correct = 0;
    float tmp;
    for (i = 0; i < count; i++)
    {
        tmp = a_data[i] + b_data[i]; // assign element i of a+b to tmp
        tmp -= c_res[i];             // compute deviation of expected and output result
        if (tmp * tmp < TOL * TOL)        // correct if square deviation is less than tolerance squared
            correct++;
        else {
            printf(" tmp %f a_data %f b_data %f c_res %f \n", tmp, a_data[i], b_data[i], c_res[i]);
        }

    }

    // summarize results
    printf("C = A+B:  %d out of %d results were correct.\n", correct, count);

    // cleanup then shutdown
    clReleaseMemObject(a_in);
    clReleaseMemObject(b_in);
    clReleaseMemObject(c_out);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

    return 0;
}

