// DetectionContourImage.cpp : Ce fichier contient la fonction 'main'. L'exécution du programme commence et se termine à cet endroit.
//
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "CL/cl.h"

const char* KernelSource =                                             "\n" \
"__kernel void mandel(                                                    \n" \
"   const double x0,                                                    \n" \
"   const double y0,                                                    \n" \
"   const double stepsize,                                                    \n" \
"   const unsigned int maxIter,                                         \n" \
"   __global unsigned int *restrict framebuffer,                        \n" \
"   const unsigned int windowWidth                                      \n" \
"   )                                                                   \n" \
"{// WORK ITEM POSITION                                                                      \n" \
"   const size_t windowPosX = get_global_id(0);                                          \n" \
"   const size_t windowPosY = get_global_id(1);                                            \n" \
"   const double stepPosX = x0 + (windowPosX * stepsize);                                           \n" \
"   const double stepPosY = y0 - (windowPosY * stepsize);                                           \n" \
"                                              \n" \
"   double x = 0.0;                                          \n" \
"   double y = 0.0;                                      \n" \
"   double x2 = 0.0;                                        \n" \
"   double y2 = 0.0;                                        \n" \
"   unsigned int i = 0;                                           \n" \
"                                              \n" \
"   while(x2 + y2 < 4.0 && i < maxIter){                                           \n" \
"        x2 = x*x;                                      \n" \
"        y2 = y*y;                                      \n" \
"        y = 2*x*y + stepPosY;                                      \n" \
"        x = x2 - y2 + stepPosX;                                      \n" \
"        i++;                          }            \n" \
"                    \n" \
"   framebuffer[windowWidth * windowPosY + windowPosX] = i%16;                                           \n" \
"}                                                                      \n" \
"\n";




void saveBMP(const char* name, int width, int height, unsigned int* data, int maxIter) {
    FILE* f;
    
    unsigned int headers[13];
    int extrabytes = 4 - ((width*3)%4);
    if (extrabytes == 4)
        extrabytes = 0;
    int paddedsize = ((width*3) + extrabytes) * height;

    headers[0] = paddedsize + 54;
    headers[1] = 0;
    headers[2] = 54;
    headers[3] = 40;
    headers[4] = width;
    headers[5] = height;

    headers[7] = 0;
    headers[8] = paddedsize;
    headers[9] = 0;
    headers[10] = 0;
    headers[11] = 0;
    headers[12] = 0;

    fopen_s(&f, name, "wb");

    int n;
    fprintf(f, "BM");
    for (n = 0; n <= 5; n++) {
        fprintf(f, "%c", headers[n] & 0x000000FF);
        fprintf(f, "%c", (headers[n] & 0x0000FF00)>>8);
        fprintf(f, "%c", (headers[n] & 0x00FF0000)>>16);
        fprintf(f, "%c", (headers[n] & (unsigned int) 0xFF000000)>>24);
    }

    fprintf(f, "%c", 1);
    fprintf(f, "%c", 0);
    fprintf(f, "%c", 24);
    fprintf(f, "%c", 0);

    for (n = 7; n <= 12; n++) {
        fprintf(f, "%c", headers[n] & 0x000000FF);
        fprintf(f, "%c", (headers[n] & 0x0000FF00) >> 8);
        fprintf(f, "%c", (headers[n] & 0x00FF0000) >> 16);
        fprintf(f, "%c", (headers[n] & (unsigned int)0xFF000000) >> 24);
    }

    int iter = -1;
    int x, y;

    for (y = height-1; y >= 0; y--) {
        for (x = 0; x < width; x++) {
            iter++;
            int mod = (unsigned char)*(data + iter);
            int r, g, b = 0;
            switch (mod)
            {
            case 0: r = 66; g = 30; b = 15; break;
            case 1: r = 25; g = 7; b = 26; break;
            case 2: r = 9; g = 1; b = 47; break;
            case 3: r = 4; g = 4; b = 73; break;
            case 4: r = 0; g = 7; b = 100; break;
            case 5: r = 12; g = 44; b = 138; break;
            case 6: r = 24; g = 82; b = 177; break;
            case 7: r = 57; g = 125; b = 209; break;
            case 8: r = 134; g = 181; b = 229; break;
            case 9: r = 211; g = 236; b = 248; break;
            case 10: r = 241; g = 233; b = 191; break;
            case 11: r = 248; g = 201; b = 95; break;
            case 12: r = 254; g = 170; b = 0; break;
            case 13: r = 204; g = 128; b = 0; break;
            case 14: r = 153; g = 87; b = 0; break;
            case 15: r = 106; g = 52; b = 3; break;
            }

            if (r < 0 || r> 255)
                r = 0;

            if (g < 0 || g> 255)
                g = 0;

            if (b < 0 || b> 255)
                b = 0;

            fprintf(f, "%c", b);
            fprintf(f, "%c", g);
            fprintf(f, "%c", r);
        }
        if (extrabytes) {
            for (n = 1; n <= extrabytes; n++) {
                fprintf(f, "%c", 0);
            }
        }
    }
    printf("DONE \n");
    fclose(f);
}

//unsigned char x2ycolor(float a, float b)


int main()
{
    int err;                   // error code returned from OpenCL calls

    size_t global;                      // global domain size  
    size_t local;                       // local  domain size  

    double step;
    int max_size, workgroup_size = 32;
    int nworkgroup;
    int i;

    cl_device_id     device_id;         // compute device id 
    cl_context       context;           // compute context
    cl_command_queue commands;          // compute command queue
    cl_program       program;           // compute program
    cl_kernel        kernel;            // compute kernel

    cl_mem y_out;                       // device memory used for the output c vector
    cl_uint numPlatforms;
    cl_platform_id firstPlatformId;

    int imgWIDTH = 1000;
    int imgHEIGHT = 1000;
    int ntotal_iter = imgHEIGHT*imgWIDTH;
    int nwork_iter = 1;
    float xmax = 1.5;
    float xmin = -2;
    float ymax = 1.75f;
    float ymin = -1.75f;

    double startX = -2;
    double startY = 1.75;
    int maxIter = 255;
    //double scale = 0.1f; step

    unsigned int* grid;
    grid = (unsigned int*)malloc(imgWIDTH * imgHEIGHT * sizeof(unsigned int));


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
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if (!context)
    {
        printf("Error: Failed to create a compute context!\n");
        return EXIT_FAILURE;
    }
    commands = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
    if (!commands)
    {
        printf("Error: Failed to create a command commands!\n");
        return EXIT_FAILURE;
    }
    program = clCreateProgramWithSource(context, 1, (const char**)&KernelSource, NULL, &err);
    if (!program)
    {
        printf("Error: Failed to create compute program!\n");
        return EXIT_FAILURE;
    }
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
    kernel = clCreateKernel(program, "mandel", &err);
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
    step = 0.0025;
    printf("Total iter : %d || Nstep : %d\n", ntotal_iter, nsteps);


    printf(" %d work groups of size %d.  %d Integration steps\n",
        (int)nworkgroup, (int)workgroup_size, nsteps);

    // Create the input (a, b) and output (c) arrays in device memory  
    y_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(unsigned int) * imgHEIGHT*imgWIDTH, NULL, NULL);
    if (!y_out)
    {
        printf("Error: Failed to allocate device memory!\n");
        exit(1);
    }

    // Set the arguments to our compute kernel
    err = 0;
    err = clSetKernelArg(kernel, 0, sizeof(cl_double), &startX);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_double), &startY);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_double), &step);
    err |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &maxIter);
    err |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &y_out);
    err |= clSetKernelArg(kernel, 5, sizeof(unsigned int), &imgWIDTH);
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
    global = nworkgroup * workgroup_size;
    local = workgroup_size;
    size_t ggg[2] = { imgWIDTH, imgHEIGHT };
    err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL, ggg, NULL, 0, NULL, &prof_event);
    if (err)
    {
        printf("Error: Failed to execute kernel!\n");
        return EXIT_FAILURE;
    }

    // Wait for the commands to complete before reading back results
    clFinish(commands);


    // Read back the results from the compute device
    err = clEnqueueReadBuffer(commands, y_out, CL_TRUE, 0, sizeof(unsigned int) * imgHEIGHT * imgWIDTH, grid, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array! %d\n", err);
        exit(1);
    }

    saveBMP("test3.bmp", imgWIDTH, imgHEIGHT, grid, maxIter);




    rtime = clock() - rtime;

    printf("\nThe kernel ran in %lf ms\n", rtime * 1000 / CLOCKS_PER_SEC);

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

    return 0;



}
