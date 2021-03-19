// ConsoleApplication3.cpp : Ce fichier contient la fonction 'main'. L'exécution du programme commence et se termine à cet endroit.
//

#include <iostream>
#include <CL/opencl.h>
#include <stdio.h>
#include <time.h>

#define DATA_SIZE 12800
#define ITERMAX 20

const char* KernelSource =
"__kernel void hello(__global double *input, __global double *output, __local double *localB)\n"\
"{\n"\
"	size_t lid = get_local_id(0);\n"\
"	size_t gid = get_group_id(0);\n"\
"	size_t gsize = get_local_size(0);\n"\
"	size_t id = lid+gid*gsize;\n"\
"	double temp = input[2*id]*input[2*id]+input[2*id+1]*input[2*id+1];\n"\
"	if(temp < 1){\n"\
"	localB[lid] = 1;}else{\n"\
"	localB[lid] = 0;}\n"\
"	\n"\
"   int i;                                          \n" \
"   double x, accu, sum;                                  \n" \
"   barrier(CLK_LOCAL_MEM_FENCE);                                 \n" \
"   sum = 0;                                                 \n" \
"   if(lid == 0){\n" \
"       for(i=0; i<gsize; i++)                                        \n" \
"           sum+=localB[i];                                             \n" \
"       output[gid] =sum;}                                             \n" \
"}\n"\
"\n";

clock_t t1, t2;

int main()
{
	t1 = clock();

	cl_context context;
	cl_context_properties properties[3];
	cl_kernel kernel;
	cl_command_queue command_queue;
	cl_program program;
	cl_int err;
	cl_platform_id platform_id;
	cl_uint num_of_platform = 0;
	cl_device_id device_id;
	cl_uint num_of_devices = 0;
	cl_mem input, output;
	size_t global;

	double inputData[DATA_SIZE * 2] = { 0 };
	double results[DATA_SIZE] = { 0 };
	double* psum_data;

	int i;

	//Init random memory, cannot be done on the GPU
	for (i = 0; i < DATA_SIZE * 2; i++) {
		inputData[i] = ((double)rand()) / RAND_MAX;
	}

	//retrieves a list of platforms available
	if (clGetPlatformIDs(1, &platform_id, &num_of_platform) != CL_SUCCESS)
	{
		printf("err unable to get platform_id \n");
		return 1;
	}

	//try to get supported GPU devices
	if (clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &num_of_devices) != CL_SUCCESS)
	{
		printf("Unable to get device_id\n");
		return 1;
	}

	//context properties list - must be terminated with 0
	properties[0] = CL_CONTEXT_PLATFORM;
	properties[1] = (cl_context_properties)platform_id;
	properties[2] = 0;

	//create a context wuth the GPU Device
	context = clCreateContext(properties, 1, &device_id, NULL, NULL, &err);

	//create a command queue using the context and device
	command_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);

	//create a program from the kernel source code
	program = clCreateProgramWithSource(context, 1, (const char**)&KernelSource, NULL, &err);

	//compile the program
	if (clBuildProgram(program, 0, NULL, NULL, NULL, NULL) != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        exit(1);
    }

	//specify which kernel from the program to execute
	kernel = clCreateKernel(program, "hello", &err);

	int nworkgroup, max_size, workgroup_size = 32;
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
	nworkgroup = DATA_SIZE / (workgroup_size * 1);

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
		workgroup_size = DATA_SIZE / (nworkgroup * 1);
	}

	psum_data = (double*)malloc(sizeof(double) * nworkgroup);

	printf(" %d work groups of size %d.  %d Integration steps\n",
		(int)nworkgroup, (int)workgroup_size, DATA_SIZE);

	//create buffers for the input and output
	input = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * DATA_SIZE * 2, NULL, NULL);
	output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(double) * nworkgroup, NULL, NULL);

	//load data into the input buffer
	clEnqueueWriteBuffer(command_queue, input, CL_TRUE, 0, sizeof(double) * DATA_SIZE * 2, inputData, 0, NULL, NULL);


	//set the argument list for the kernel command
	clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);
	clSetKernelArg(kernel, 2, sizeof(double)* workgroup_size, NULL);
	global = DATA_SIZE;
	size_t local = workgroup_size;

	int j;
	double rf = 0;
	double piFinal = 0;
	double otimeSum = 0;
	cl_event prof_event;
	for (j = 0; j < ITERMAX; j++) {
		//Init random memory, cannot be done on the GPU
		for (i = 0; i < DATA_SIZE * 2; i++) {
			inputData[i] = ((double)rand()) / RAND_MAX;
		}

		//enqueue the kernel command for execution
		clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global, &local, 0, NULL, &prof_event);

		//wait for queue to empty
		clFinish(command_queue);

		//copy the result from out of the output buffer
		clEnqueueReadBuffer(command_queue, output, CL_TRUE, 0, sizeof(double) * nworkgroup, psum_data, 0, NULL, NULL);

		for (i = 0;i < nworkgroup;i++) {
			rf += psum_data[i];
		}
		rf /= DATA_SIZE;
		rf *= 4;

		piFinal += rf;

		// extract timing data from the event, prof_event
		err = clWaitForEvents(1, &prof_event);
		cl_ulong ev_start_time = (cl_ulong)0;
		cl_ulong ev_end_time = (cl_ulong)0;
		err = clGetEventProfilingInfo(prof_event, CL_PROFILING_COMMAND_START,
			sizeof(cl_ulong), &ev_start_time, NULL);
		err = clGetEventProfilingInfo(prof_event, CL_PROFILING_COMMAND_END,
			sizeof(cl_ulong), &ev_end_time, NULL);
		otimeSum += (double)(ev_end_time - ev_start_time) * 1.0e-6;
	}

	piFinal /= ITERMAX;
		
	

	

	//cleanup - release OpenCL ressources
	clReleaseMemObject(input);
	clReleaseMemObject(output);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);


	


	t2 = clock();
	printf("\n ==== PAR MODE ==== \n");
	printf("For %d iterations :\n", DATA_SIZE* ITERMAX);
	printf("Pi final : %g\n ", piFinal);
	printf("Total Time OpenCL %.3lfms\n", (((double)t2 - t1) / (double)CLOCKS_PER_SEC) * 1000);

	printf("prof says %f ms \n", otimeSum);

	//seq mode
	t1 = clock();

	rf = 0;
	int jseq;

	for (jseq = 0; jseq < ITERMAX; jseq++) {
		for (i = 0; i < DATA_SIZE * 2; i++) {
			inputData[i] = ((double)rand()) / RAND_MAX;
		}

		double inc = 0;
		for (i = 0; i < DATA_SIZE; i++) {
			double temp0 = inputData[2 * i] * inputData[2 * i] + inputData[2 * i + 1] * inputData[2 * i + 1];
			if (temp0 < 1) {
				inc += 1;
			}
		}

		//compute the results
		rf += 4 * inc / DATA_SIZE;
	}

	rf /= ITERMAX;

	t2 = clock();
	printf("\n ==== SEQ MODE ==== \n");
	printf("For %d iterations :\n", DATA_SIZE* ITERMAX);
	printf("Pi final : %g\n ", rf);
	printf("Total Time Seq %.3lfms\n", (((double)t2 - t1) / (double)CLOCKS_PER_SEC) * 1000);

	return 0;
}