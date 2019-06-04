#include "AssignmentFunctions.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <algorithm> 

#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

//Function takes a vector and splits it into two evenly sized halfs
std::vector<std::vector<float>> splitData(std::vector<float> inputVec, float padNum) {
	//left assigned to first half of data
	std::vector<float> left(inputVec.begin(), inputVec.begin() + inputVec.size() / 2);
	//right assigned to second half of data
	std::vector<float> right(inputVec.begin() + inputVec.size() / 2, inputVec.end());

	//Make each half even by padding one side
	if (left.size() < right.size()) 
		left.push_back(padNum);
	else if (right.size() < left.size()) 
		right.push_back(padNum);

	std::vector<std::vector<float>> out = { left, right };
	return out;
}

float findMean(std::vector<float> inputVec, cl::Context context, cl::CommandQueue queue, cl::Program program)
{
	//Calculate Average
	std::cout << "\n----------\nCALCULATING AVERAGE\n----------\n";
	//Memory alocation
	std::vector<std::vector<float>> inputVecSplit = splitData(inputVec, 0);
	//split the data and assign left side to A and right side to B
	std::vector<float> A = inputVecSplit[0];
	std::vector<float> B = inputVecSplit[1];
	size_t vector_elements = A.size();//number of elements
	size_t vector_size = A.size() * sizeof(float);//size in bytes
	std::vector<float> C(vector_elements);
	//this loop takes each half and adds them together in parallel
	//the result is then split in half and the loop is repeated
	while (C.size() > 0)
	{
		//size of A changes each loop
		vector_elements = A.size();//number of elements
		vector_size = A.size() * sizeof(float);//size in bytes

		//device - buffers
		cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, vector_size);
		cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, vector_size);
		cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, vector_size);

		//Copy arrays A and B to device memory
		cl::Event A_event;
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, vector_size, &A[0], NULL, &A_event);
		cl::Event B_event;
		queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, vector_size, &B[0], NULL, &B_event);

		std::cout << "Copy A to device memory time [ns]:" <<
			(A_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
				A_event.getProfilingInfo<CL_PROFILING_COMMAND_START>()) << std::endl;
		std::cout << "Copy B to device memory time [ns]:" <<
			(B_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
				B_event.getProfilingInfo<CL_PROFILING_COMMAND_START>()) << std::endl;

		//Setup and execute the kernel (i.e. device code)
		cl::Kernel kernel_add = cl::Kernel(program, "add");
		kernel_add.setArg(0, buffer_A);
		kernel_add.setArg(1, buffer_B);
		kernel_add.setArg(2, buffer_C);

		cl::Event kernel_add_event;
		queue.enqueueNDRangeKernel(kernel_add, cl::NullRange,
			cl::NDRange(vector_elements), cl::NullRange, NULL, &kernel_add_event);

		//Copy the result from device to host
		cl::Event C_event;
		queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, vector_size, &C[0], NULL, &C_event);

		std::cout << "Copy C to host time [ns]:" <<
			(C_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
				C_event.getProfilingInfo<CL_PROFILING_COMMAND_START>()) << std::endl;

		std::cout << "Kernel execution time [ms]:" << (kernel_add_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - kernel_add_event.getProfilingInfo<CL_PROFILING_COMMAND_START>()) << "\n" << std::endl;

		//if we have a single value, stop the loop
		if (C.size() == 1)
			break;

		//split the output and assign left to A and right to B
		inputVecSplit = splitData(C, 0);
		A = inputVecSplit[0];
		B = inputVecSplit[1];
		C.resize(A.size());
	}
	//return sum of values / num of values
	return (float)C[0] / (float)inputVec.size();
}

float findMin(std::vector<float> inputVec, cl::Context context, cl::CommandQueue queue, cl::Program program) {
	//Calculate Low
	std::cout << "\n----------\nCALCULATING LOWEST VALUE\n----------\n";
	//Memory alocation (pad with extreme high value so that padding isn't accidently selected as low)
	std::vector<std::vector<float>> inputVecSplit = splitData(inputVec, 99999);
	//split the data and assign left side to A and right side to B
	std::vector<float> A = inputVecSplit[0];
	std::vector<float> B = inputVecSplit[1];
	size_t vector_elements = A.size();//number of elements
	size_t vector_size = A.size() * sizeof(float);//size in bytes
	std::vector<float> C(vector_elements);

	while (C.size() > 0)
	{
		//size of A changes each loop
		vector_elements = A.size();//number of elements
		vector_size = A.size() * sizeof(float);//size in bytes

		//device - buffers
		cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, vector_size);
		cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, vector_size);
		cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, vector_size);

		//Copy arrays A and B to device memory
		cl::Event A_event;
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, vector_size, &A[0], NULL, &A_event);
		cl::Event B_event;
		queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, vector_size, &B[0], NULL, &B_event);

		std::cout << "Copy A to device memory time [ns]:" <<
			(A_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
				A_event.getProfilingInfo<CL_PROFILING_COMMAND_START>()) << std::endl;
		std::cout << "Copy B to device memory time [ns]:" <<
			(B_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
				B_event.getProfilingInfo<CL_PROFILING_COMMAND_START>()) << std::endl;

		//Setup and execute the kernel (i.e. device code)
		cl::Kernel kernel_getLow = cl::Kernel(program, "getLow");
		kernel_getLow.setArg(0, buffer_A);
		kernel_getLow.setArg(1, buffer_B);
		kernel_getLow.setArg(2, buffer_C);

		cl::Event kernel_getLow_event;
		queue.enqueueNDRangeKernel(kernel_getLow, cl::NullRange,
			cl::NDRange(vector_elements), cl::NullRange, NULL, &kernel_getLow_event);

		//Copy the result from device to host
		cl::Event C_event;
		queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, vector_size, &C[0], NULL, &C_event);

		std::cout << "Copy C to host time [ns]:" <<
			(C_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
				C_event.getProfilingInfo<CL_PROFILING_COMMAND_START>()) << std::endl;

		std::cout << "Kernel execution time [ms]:" << (kernel_getLow_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - kernel_getLow_event.getProfilingInfo<CL_PROFILING_COMMAND_START>()) << "\n" << std::endl;

		//if we have a single value, stop the loop
		if (C.size() == 1)
			break;

		//split the output and assign left to A and right to B
		inputVecSplit = splitData(C, 99999);
		A = inputVecSplit[0];
		B = inputVecSplit[1];
		C.resize(A.size());
	}
	return (float)C[0];
}

float findMax(std::vector<float> inputVec, cl::Context context, cl::CommandQueue queue, cl::Program program) {
	//Calculate High
	std::cout << "\n----------\nCALCULATING HIGHEST VALUE\n----------\n";
	//Memory alocation (pad with extreme low value so that padding is accidently selected as high)
	std::vector<std::vector<float>> inputVecSplit = splitData(inputVec, -99999);
	//split the data and assign left side to A and right side to B
	std::vector<float> A = inputVecSplit[0];
	std::vector<float> B = inputVecSplit[1];
	size_t vector_elements = A.size();//number of elements
	size_t vector_size = A.size() * sizeof(float);//size in bytes
	std::vector<float> C(vector_elements);

	while (C.size() > 0)
	{
		//size of A changes each loop
		vector_elements = A.size();//number of elements
		vector_size = A.size() * sizeof(float);//size in bytes

		//device - buffers
		cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, vector_size);
		cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, vector_size);
		cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, vector_size);

		//Copy arrays A and B to device memory
		cl::Event A_event;
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, vector_size, &A[0], NULL, &A_event);
		cl::Event B_event;
		queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, vector_size, &B[0], NULL, &B_event);

		std::cout << "Copy A to device memory time [ns]:" <<
			(A_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
				A_event.getProfilingInfo<CL_PROFILING_COMMAND_START>()) << std::endl;
		std::cout << "Copy B to device memory time [ns]:" <<
			(B_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
				B_event.getProfilingInfo<CL_PROFILING_COMMAND_START>()) << std::endl;

		//Setup and execute the kernel (i.e. device code)
		cl::Kernel kernel_getHigh = cl::Kernel(program, "getHigh");
		kernel_getHigh.setArg(0, buffer_A);
		kernel_getHigh.setArg(1, buffer_B);
		kernel_getHigh.setArg(2, buffer_C);

		cl::Event kernel_getHigh_event;
		queue.enqueueNDRangeKernel(kernel_getHigh, cl::NullRange,
			cl::NDRange(vector_elements), cl::NullRange, NULL, &kernel_getHigh_event);

		//Copy the result from device to host
		cl::Event C_event;
		queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, vector_size, &C[0], NULL, &C_event);

		std::cout << "Copy C to host time [ns]:" <<
			(C_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
				C_event.getProfilingInfo<CL_PROFILING_COMMAND_START>()) << std::endl;

		std::cout << "Kernel execution time [ms]:" << (kernel_getHigh_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - kernel_getHigh_event.getProfilingInfo<CL_PROFILING_COMMAND_START>()) << "\n" << std::endl;

		//if we have a single value, stop the loop
		if (C.size() == 1)
			break;

		//split the output and assign left to A and right to B
		inputVecSplit = splitData(C, -9999999);
		A = inputVecSplit[0];
		B = inputVecSplit[1];
		C.resize(A.size());
	}
	return (float)C[0];
}

float findStdDev(std::vector<float> inputVec, float average , cl::Context context, cl::CommandQueue queue, cl::Program program)
{
	//Calculate High
	std::cout << "\n----------\nCALCULATING STD DEV\n----------\n";

	//Memory Allocation
	std::vector<float> A = inputVec;
	std::vector<float> B(A.size());
	std::fill(B.begin(), B.end(), average);//fill each space in B with calculated average
	size_t vector_elements = A.size();//number of elements
	size_t vector_size = A.size() * sizeof(float);//size in bytes
	std::vector<float> C(vector_elements);

	//device - buffers
	cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, vector_size);
	cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, vector_size);
	cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, vector_size);

	//Copy arrays A and B to device memory
	cl::Event A_event;
	queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, vector_size, &A[0], NULL, &A_event);
	cl::Event B_event;
	queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, vector_size, &B[0], NULL, &B_event);

	std::cout << "Copy A to device memory time [ns]:" <<
		(A_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
			A_event.getProfilingInfo<CL_PROFILING_COMMAND_START>()) << std::endl;
	std::cout << "Copy B to device memory time [ns]:" <<
		(B_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
			B_event.getProfilingInfo<CL_PROFILING_COMMAND_START>()) << std::endl;

	//Setup and execute the kernel (i.e. device code)
	cl::Kernel kernel_getStdDev = cl::Kernel(program, "getStdDev");
	kernel_getStdDev.setArg(0, buffer_A);
	kernel_getStdDev.setArg(1, buffer_B);
	kernel_getStdDev.setArg(2, buffer_C);

	cl::Event kernel_getStdDev_event;
	queue.enqueueNDRangeKernel(kernel_getStdDev, cl::NullRange,
		cl::NDRange(vector_elements), cl::NullRange, NULL, &kernel_getStdDev_event);

	//Copy the result from device to host
	cl::Event C_event;
	queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, vector_size, &C[0], NULL, &C_event);

	std::cout << "Copy C to host time [ns]:" <<
		(C_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
			C_event.getProfilingInfo<CL_PROFILING_COMMAND_START>()) << std::endl;

	std::cout << "Kernel execution time [ms]:" << (kernel_getStdDev_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - kernel_getStdDev_event.getProfilingInfo<CL_PROFILING_COMMAND_START>()) << "\n" << std::endl;
	
	//average the squared difference then square root
	return sqrt(findMean(C, context, queue, program));
}