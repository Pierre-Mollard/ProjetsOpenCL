// ConsoleApplication3.cpp : Ce fichier contient la fonction 'main'. L'exécution du programme commence et se termine à cet endroit.
//

#include <iostream>
#include <CL/opencl.h>
#include <stdio.h>
#include <time.h>

#define DATA_SIZE 128
#define ITER_MAX 2000

const char* KernelSource =
"__kernel void hello(__global double *input, __global double *output)\n"\
"{\n"\
"	size_t id = get_global_id(0);\n"\
"	size_t group_size = get_global_size(0);\n"\
"	double temp = input[2*id]*input[2*id]+input[2*id+1]*input[2*id+1];\n"\
"	if(temp < 1){\n"\
"	output[id] = 1;}else{\n"\
"	output[id] = 0;}\n"\
"	\n"\
"	for(uint stride=group_size/2; stride>0; stride/=2){\n"\
"	barrier(CLK_LOCAL_MEM_FENCE);\n"\
"	if(id < stride)\n"\
"	output[id] += output[id+stride];}\n"\
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

	double inputData[DATA_SIZE*2] = { 0 };
	double results[DATA_SIZE] = { 0 };

	int i;

	//Init random memory, cannot be done on the GPU
	for (i = 0; i < DATA_SIZE*2; i++) {
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
	command_queue = clCreateCommandQueue(context, device_id, 0, &err);

	//create a program from the kernel source code
	program = clCreateProgramWithSource(context, 1, (const char**) &KernelSource, NULL, &err);

	//compile the program
	if (clBuildProgram(program, 0, NULL, NULL, NULL, NULL) != CL_SUCCESS) 
	{
		printf("Err unable to build program\n");
		return 1;
	}

	//specify which kernel from the program to execute
	kernel = clCreateKernel(program, "hello", &err);

	//create buffers for the input and output
	input = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * DATA_SIZE * 2, NULL, NULL);
	output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(double), NULL, NULL);

	//load data into the input buffer
	clEnqueueWriteBuffer(command_queue, input, CL_TRUE, 0, sizeof(double) * DATA_SIZE * 2, inputData, 0, NULL, NULL);

	//set the argument list for the kernel command
	clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);
	global = DATA_SIZE;

	int iter;
	double r2[ITER_MAX];
	double rf = 0;
	double temp;
	for (iter = 0; iter < ITER_MAX-1; iter++) {

		//enqueue the kernel command for execution
		clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global, NULL, 0, NULL, NULL);

		//create new random numbers for the next iteration (while gpu is sync up)
		for (i = 0; i < DATA_SIZE * 2; i++) {
			inputData[i] = ((double)rand()) / RAND_MAX;
		}

		//wait for queue to empty
		clFinish(command_queue);

		//copy the result from out of the output buffer
		clEnqueueReadBuffer(command_queue, output, CL_TRUE, 0, sizeof(double), results, 0, NULL, NULL);

		//get the result
		temp = results[0];
		r2[iter] = 4 *temp / DATA_SIZE;
		rf += r2[iter];


	}
	
	//average the the results
	rf /= ITER_MAX;

	//cleanup - release OpenCL ressources
	clReleaseMemObject(input);
	clReleaseMemObject(output);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);

	t2 = clock();
	printf("\n ==== PAR MODE ==== \n");
	printf("For %d iterations :\n", DATA_SIZE * ITER_MAX);
	printf("Pi final : %g\n ", rf);
	printf("Total Time OpenCL %.3lfms\n", ((double)(t2 - t1) / (double)CLOCKS_PER_SEC)*1000);

	//seq mode
	t1 = clock();

	rf = 0;
	for (iter = 0; iter < ITER_MAX; iter++) {

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
		r2[iter] = 4 * inc / DATA_SIZE;
		rf += r2[iter];


	}

	//average the results
	rf /= ITER_MAX;

	t2 = clock();
	printf("\n ==== SEQ MODE ==== \n");
	printf("For %d iterations :\n", DATA_SIZE * ITER_MAX);
	printf("Pi final : %g\n ", rf);
	printf("Total Time Seq %.3lfms\n", ((double)(t2 - t1) / (double)CLOCKS_PER_SEC)*1000);

	return 0;
}