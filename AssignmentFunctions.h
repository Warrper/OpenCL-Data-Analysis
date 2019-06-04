#pragma once

#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include <vector>
#include <iostream>

std::vector<std::vector<float>> splitData(std::vector<float> inputVec, float padNum);

float findMean(std::vector<float> inputVec, cl::Context context, cl::CommandQueue queue, cl::Program program);

float findMin(std::vector<float> inputVec, cl::Context context, cl::CommandQueue queue, cl::Program program);

float findMax(std::vector<float> inputVec, cl::Context context, cl::CommandQueue queue, cl::Program program);

float findStdDev(std::vector<float> inputVec, float average, cl::Context context, cl::CommandQueue queue, cl::Program program);