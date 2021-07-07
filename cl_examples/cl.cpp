#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <time.h>

#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cstring>
#include <math.h>

// const char* kernel_matrix =
// 		"__kernel void multiplication(__global const float* left, 	\n"
// 		"						 __global const float* right, 		\n"
// 		"						 __global float* result,			\n"
//         "                                              int N) 	    \n"
// 		"{ 															\n"
// 		"	int row = get_global_id(0); 							\n"
// 		"	int col = get_global_id(1);								\n"
// 		"	float temp = 0; 										\n"
// 		"	for (int i = 0; i < N; i++) { 							\n"
// 		"		temp += left[row * N + i] * right[i * N + col]; 	\n"
// 		"	}														\n"
// 		"	result[row * N + col] = temp;							\n"
// 		"}															\n";

const char* kernel_matrix =
		"__kernel void multiplication(__global const float* left, 	\n"
		"						 __global const float* right, 		\n"
		"						 __global float* result,			\n"
        "                                              int N) 	    \n"
		"{ 															\n"
		"	int id = get_global_id(0); 							\n"
		"	int row = id / N;										\n"
		"	int col = id % N; 										\n"
		"	float temp = 0; 										\n"
		"	for (int i = 0; i < N; i++) { 							\n"
		"		temp += left[row * N + i] * right[i * N + col]; 	\n"
		"	}														\n"
		"	result[id] = temp;							\n"
		"}															\n";

const char* kernel_addition =
		"__kernel void addition(__global const float* left, 	\n"
		"						 __global const float* right, 	\n"
		"						 __global float* result, 		\n"
		"										int n)			\n"
		"{ 														\n"
		"	int id = get_global_id(0); 							\n"
		"	if (id < n) {										\n"
		"		result[id] = left[id] + right[id]; 				\n"
		"	}													\n"
		"}														\n";

float* generate(int row, int col, bool tag) {

	float* arr = (float*)malloc(row * col * sizeof(float));

	int total = row * col;
	srand(time(NULL));

	for (int i = 0; i < total; i++) {

		if (tag) {
			arr[i] = 0;
		} else {
			arr[i] = (float) rand() / RAND_MAX * 0.01;
		}
	}
	return arr;
}

void execute_multiplication(float* left, float* right, float* result, int M, clock_t& warmup, clock_t& compilation, clock_t& link, clock_t& copy, clock_t& calculation, clock_t& execution);

void execute_addition(float* left, float* right, float* result, int nums,
		clock_t& warmup, clock_t& compilation, clock_t& link, clock_t& copy, clock_t& calculation, clock_t& execution);

void validate_addition(float*, float*, float*, int);

void validate_matrix(float*, float*, float*, int);

void print(float* arr, int row, int col) {
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      std::cout << arr[i * col + j] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << "----------------" << std::endl;
}

int main(int argc, char* argv[]) {

	std::string first = argv[1];
	std::string second = argv[2];
	size_t first_pos, second_pos;

	int M = std::stoi(first, &first_pos);
	int epoch = std::stoi(second, &second_pos);

	int N = 10000;

	float* left = generate(M, N, false);
	float* right = generate(M, N, false);
	float* result = generate(M, N, true);
	std::cout << "generated data" << std::endl;

	clock_t warmup = 0;
	clock_t compilation = 0;
	clock_t link = 0;
	clock_t copy = 0;
	clock_t calculation = 0;
	clock_t execution = 0;

	for (int i = 0; i < epoch; i++) {
		execute_addition(left, right, result, M*N, warmup, compilation, link, copy, calculation, execution);
		std::cout << "count[" << i << "]" << std::endl;
	}

	std::cout << "timing on warmup: " << warmup / epoch << std::endl;
	std::cout << "timing on compilation: " << compilation / epoch << std::endl;
	std::cout << "timing on link: " << link / epoch << std::endl;
	std::cout << "timing on copy: " << copy / epoch << std::endl;
	std::cout << "timing on total calculation: " << calculation / epoch << std::endl;
	std::cout << "timing on execution: " << execution / epoch << std::endl;
	
//	print(result, M, N);

	validate_addition(left, right, result, M*N);

	free(left);
	free(right);
	free(result);
}



void validate_addition(float* left, float* right, float* result, int N) {
		std::cout << "validating..." << std::endl;
	int count = 0;

	for (int i = 0; i < N; i++) {
		float temp = left[i] + right[i];
		
		float diff = abs(temp - result[i]);

		if (diff > 2e-7) {
			count += 1;
			std::cout << "error" << std::endl;
		}
	}

	if (count == 0) {
		std::cout << "successfully" <<std::endl;
	} else {
		std::cout << "someing wrong: " << count << std::endl;
	}

}


	
void validate_matrix(float* left, float* right, float* result, int N) {
	std::cout << "validating..." << std::endl;
	int count = 0;

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			float temp = 0;

			
			for (int k = 0; k < N; k++) {
				temp += left[i * N + k] * right[k * N + j];
			}
			// std::cout << temp << " ";
			float diff = abs(temp - result[i * N + j]);
			if (diff > 2e-7) {
				count += 1;
				std::cout << "diff: " << diff << std::endl;
			}
		}
		// std::cout << std::endl;
	}
	
	if (count == 0) {
		std::cout << "successfully" <<std::endl;
	} else {
		std::cout << "someing wrong: " << count << std::endl;
	}
}

void execute_multiplication(float* left, float* right, float* result, int M, clock_t& warmup, clock_t& compilation, clock_t& link, clock_t& copy, clock_t& calculation, clock_t& execution) {

	// size_t global_size[2], local_size[2];
	// local_size[0] = 64;
	// local_size[1] = 64;
	// global_size[0] = M;
	// global_size[1] = M;

	size_t local_size, global_size;
	local_size = 64;
	global_size = ceil(M * M/(float)local_size) * local_size;

	size_t bytes_left = M * M * sizeof(float);
	size_t bytes_right = M * M * sizeof(float);
	size_t bytes_result = M * M * sizeof(float);

	size_t kernel_size = strlen(kernel_matrix);

	auto start = clock();
	// OpenCL memory object
	cl_mem acc_left;
	cl_mem acc_right;
	cl_mem acc_result;
	// OpenCL object
	cl_platform_id platform;
	cl_device_id device;
	cl_uint device_nums = 0;
	cl_context context;
	cl_command_queue queue;
	cl_program program;
	cl_kernel kernel;

	cl_int err;
	err = clGetPlatformIDs(1, &platform, NULL);
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &device_nums);
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, device_nums, &device, NULL);

//	char str[256];
//	clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(str) - 1, str, NULL);
//	std::cout << "device name: " << str << std::endl;

	context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);

	queue = clCreateCommandQueue(context, device, 0, &err);

	auto end_warmup = clock();

	program = clCreateProgramWithSource(context, 1, &kernel_matrix, &kernel_size, &err);
	
	err = clCompileProgram(program, device_nums, &device, NULL, 0, NULL, NULL, NULL, NULL);
//	std::cout << "compile: " << err << std::endl;

	auto end_compilation = clock();

	program = clLinkProgram(context, device_nums, &device, NULL, 1, &program, NULL, NULL, &err);
//	std::cout << "link: " << err << std::endl;



//	clBuildProgram(program, 1, &device, NULL, NULL, NULL);

	kernel = clCreateKernel(program, "multiplication", &err);
//	std::cout << "kernel: " << err << std::endl;

	auto end_link = clock();


	acc_left = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes_left, NULL, &err);
	acc_right = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes_right, NULL, &err);
	acc_result = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes_result, NULL, &err);

	
	err = clEnqueueWriteBuffer(queue, acc_left, CL_TRUE, 0, bytes_left, left, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(queue, acc_right, CL_TRUE, 0, bytes_right, right, 0, NULL, NULL);
//	std::cout << "write buffer: " << err << std::endl;


	auto end_copy = clock();

	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &acc_left);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &acc_right);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &acc_result);
	err |= clSetKernelArg(kernel, 3, sizeof(int), &M);


	err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
	std::cout << "execution: " << err << std::endl;

	clFinish(queue);

	auto end_calculation = clock();

	clEnqueueReadBuffer(queue, acc_result, CL_TRUE, 0, bytes_result, result, 0, NULL, NULL);


	clReleaseMemObject(acc_left);
	clReleaseMemObject(acc_right);
	clReleaseMemObject(acc_result);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);

	auto end_execution = clock();

	warmup += end_warmup - start;
	compilation += end_compilation - end_warmup;
	link += end_link - end_compilation;
	copy += end_copy - end_link;
	calculation += end_calculation - end_copy;
	execution += end_execution - start;
}

void execute_addition(float* left, float* right, float* result, int nums,
		clock_t& warmup, clock_t& compilation, clock_t& link, clock_t& copy, clock_t& calculation, clock_t& execution) {

	size_t global_size, local_size;
	local_size = 64;
	global_size = ceil(nums/(float)local_size) * local_size;
	size_t bytes = nums * sizeof(float);
	size_t kernel_size = strlen(kernel_addition);

	auto start = clock();
	// OpenCL memory object
	cl_mem acc_left;
	cl_mem acc_right;
	cl_mem acc_result;
	// OpenCL object
	cl_platform_id platform;
	cl_device_id device;
	cl_uint device_nums = 0;
	cl_context context;
	cl_command_queue queue;
	cl_program program;
	cl_kernel kernel;

	cl_int err;
	err = clGetPlatformIDs(1, &platform, NULL);
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &device_nums);
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, device_nums, &device, NULL);

//	char str[256];
//	clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(str) - 1, str, NULL);
//	std::cout << "device name: " << str << std::endl;

	context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
	queue = clCreateCommandQueue(context, device, 0, &err);

	auto end_warmup = clock();

	program = clCreateProgramWithSource(context, 1, &kernel_addition, &kernel_size, &err);
	err = clCompileProgram(program, device_nums, &device, NULL, 0, NULL, NULL, NULL, NULL);
	auto end_compile = clock();
	program = clLinkProgram(context, device_nums, &device, NULL, 1, &program, NULL, NULL, &err);
	auto end_link = clock();

//	clBuildProgram(program, 1, &device, NULL, NULL, NULL);

	kernel = clCreateKernel(program, "addition", &err);

	auto end_creation = clock();

	acc_left = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
	acc_right = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
	acc_result = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, NULL);

	err = clEnqueueWriteBuffer(queue, acc_left, CL_TRUE, 0, bytes, left, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(queue, acc_right, CL_TRUE, 0, bytes, right, 0, NULL, NULL);
	auto end_copy = clock();

	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &acc_left);
	err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &acc_right);
	err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &acc_result);
	err = clSetKernelArg(kernel, 3, sizeof(int), &nums);

	err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
	clFinish(queue);

	auto end_calculation = clock();

	clEnqueueReadBuffer(queue, acc_result, CL_TRUE, 0, bytes, result, 0, NULL, NULL);


	clReleaseMemObject(acc_left);
	clReleaseMemObject(acc_right);
	clReleaseMemObject(acc_result);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);

	auto end_execution = clock();

	warmup += end_warmup - start;
	compilation += end_compile - end_warmup;
	link += end_link - end_compile;
	copy += end_copy - end_link;
	calculation += end_calculation - end_copy;
	execution += end_execution - start;
}