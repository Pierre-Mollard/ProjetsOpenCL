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

#define NWORKITER (100000)    // number of iters per work item
#define NTOTALITER (256*256*256)    // number of total iter

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
"__kernel void pi_inte(                                                         \n" \
"   __local double* ylocal,                                                  \n" \
"   __global double* ypartial,                                                  \n" \
"   const double step,                                                  \n" \
"   const unsigned int nworkiter)                                           \n" \
"{                                                                      \n" \
"   int localID = get_local_id(0);                                           \n" \
"   int n_workitems = get_local_size(0);                                           \n" \
"   int groupID = get_group_id(0);                                           \n" \
"   int ibegin = (localID + groupID*n_workitems)*nworkiter;                         \n" \
"   int iend = ibegin + nworkiter;                                          \n" \
"   int i;                                          \n" \
"   double x, accu, sum;                                          \n" \
"   for(i=ibegin; i<iend; i++){                                        \n" \
"       x=(i+0.5)*step;                                              \n" \
"       accu+=4.0/(1+(x*x)); }                                            \n" \
"   ylocal[localID]=accu;                                \n" \
"   barrier(CLK_LOCAL_MEM_FENCE);                                 \n" \
"   sum = 0;                                                 \n" \
"   if(localID == 0){\n" \
"       for(i=0; i<n_workitems; i++)                                        \n" \
"           sum+=ylocal[i];                                             \n" \
"       ypartial[groupID] =sum;                                             \n" \
"   }                                      \n" \
"}                                                                      \n" \
"\n";

//------------------------------------------------------------------------------


int main(int argc, char** argv)
{
    int          err;                   // error code returned from OpenCL calls

    size_t global;                      // global domain size  
    size_t local;                       // local  domain size  
    
    int ntotal_iter = NTOTALITER;
    int nwork_iter = NWORKITER;
    double step;
    int max_size, workgroup_size = 32;
    int nworkgroup;
    int i;

    double        *psum_data;        // c vector (a+b) returned from the compute device
    
    cl_device_id     device_id;         // compute device id 
    cl_context       context;           // compute context
    cl_command_queue commands;          // compute command queue
    cl_program       program;           // compute program
    cl_kernel        kernel;            // compute kernel

    cl_mem y_out;                       // device memory used for the output c vector

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
    kernel = clCreateKernel(program, "pi_inte", &err);
    if (!kernel || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel!\n");
        exit(1);
    }

    // Get the maximum work group size for executing the kernel on the device
    err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE,
        sizeof(max_size), &max_size, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to retrieve kernel work group info! %d\n", err);
        exit(1);
    }
    if (max_size > workgroup_size) workgroup_size = max_size;

    // Now that we know the size of the work_groups, we can set the number of work
    // groups, the actual number of steps, and the step size
    nworkgroup = ntotal_iter / (workgroup_size * nwork_iter);

    if (nworkgroup < 1)
    {
        int comp_units;
        err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &comp_units, NULL);
        if (err != CL_SUCCESS)
        {
            printf("Error: Failed to access device number of compute units !\n");
            return EXIT_FAILURE;
        }
        nworkgroup = comp_units;
        workgroup_size = ntotal_iter / (nworkgroup * nwork_iter);
    }
    int nsteps = workgroup_size * nwork_iter * nworkgroup;
    step = 1.0 / (double)nsteps;

    printf("Total iter : %d || Nstep : %d\n", ntotal_iter, nsteps);
    
    psum_data = (double*)malloc(sizeof(double) * nworkgroup);


    printf(" %d work groups of size %d.  %d Integration steps\n",
        (int)nworkgroup, (int)workgroup_size, nsteps);

    // Create the input (a, b) and output (c) arrays in device memory  
    y_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(double) * nworkgroup, NULL, NULL);
    if (!y_out)
    {
        printf("Error: Failed to allocate device memory!\n");
        exit(1);
    }

    // Set the arguments to our compute kernel
    err = 0;
    err = clSetKernelArg(kernel, 0, sizeof(double) * workgroup_size, NULL);
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &y_out);
    err |= clSetKernelArg(kernel, 2, sizeof(double), &step);
    err |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &nwork_iter);
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
    global = nworkgroup*workgroup_size;
    local = workgroup_size;
    err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &local, 0, NULL, &prof_event);
    if (err)
    {
        printf("Error: Failed to execute kernel!\n");
        return EXIT_FAILURE;
    }

    // Wait for the commands to complete before reading back results
    clFinish(commands);
   

    // Read back the results from the compute device
    err = clEnqueueReadBuffer(commands, y_out, CL_TRUE, 0, sizeof(double) * nworkgroup, psum_data, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array! %d\n", err);
        exit(1);
    }

    // Test the results
    double res = 0.0;
    for (i = 0; i < nworkgroup; i++)
    {
        res += psum_data[i];
       

    }

    res *= step;

    // summarize results
    printf("Pi = %lf\n", res);

    rtime = clock() - rtime;

    printf("\nThe kernel ran in %lf ms\n", rtime*1000/CLOCKS_PER_SEC);

    // extract timing data from the event, prof_event
    err = clWaitForEvents(1, &prof_event);
    cl_ulong ev_start_time = (cl_ulong)0;
    cl_ulong ev_end_time = (cl_ulong)0;
    err = clGetEventProfilingInfo(prof_event, CL_PROFILING_COMMAND_START,
        sizeof(cl_ulong), &ev_start_time, NULL);
    err = clGetEventProfilingInfo(prof_event, CL_PROFILING_COMMAND_END,
        sizeof(cl_ulong), &ev_end_time, NULL);
    printf("prof says %f ms \n", (double)(ev_end_time - ev_start_time) * 1.0e-6);



    // cleanup then shutdown
    clReleaseMemObject(y_out);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

    /// SEQ
    rtime = 0.0;
    rtime = clock();
    double x = 0;
    double sum = 0;
    for (i = 0;i < nsteps;i++) {
        x = (i + 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }
    sum *= step;
    rtime = clock() - rtime;

    printf("SEQ : PI=%g || Time : %g ms\n", sum, rtime*1000/CLOCKS_PER_SEC);


    return 0;
}

