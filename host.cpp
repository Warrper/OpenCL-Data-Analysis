#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include "Utils.h"
#include "AssignmentFunctions.h"

void print_help() {
	std::cerr << "Application usage:" << std::endl;
	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

int platform_id = 0;
int device_id = 0;

int main(int argc, char **argv) {
	
	float avgTemp, lowTemp, highTemp, stdDevTemp;

	//Handle command line options such as device selection, verbosity, etc.
	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0; }
	}

	//Read in data
	std::ifstream myfile("src/temp_lincolnshire.txt");
	std::string name, year, month, day, time;
	float temp;
	std::vector<float> temps;
	while (myfile >> name >> year >> month >> day >> time >> temp){
		//printf("name: %s, temp: %f\n", name.c_str(), temp);
		//Read temp into a vector
		temps.push_back(temp);
	}

	try{
		//Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		//display the selected device
		std::cout << "Running on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		//create a queue to which we will push commands for the device
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);

		//Load & build the device code
		cl::Program::Sources sources;
		AddSources(sources, "src/device.cl");
		cl::Program program(context, sources);

		//build and debug the kernel code
		try {
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}
		avgTemp = findMean(temps, context, queue, program);
		lowTemp = findMin(temps, context, queue, program);
		highTemp = findMax(temps, context, queue, program);
		stdDevTemp = findStdDev(temps, avgTemp, context, queue, program);

		std::cout <<  "Avg Temp = " << avgTemp << std::endl;
		std::cout << "Std Dev = " << stdDevTemp << std::endl;
		std::cout << "Low Temp = " << lowTemp << std::endl;
		std::cout << "High Temp = " << highTemp << std::endl;

		std::cout << "\nPress Enter to Exit..."; std::getchar();
	}
	catch (cl::Error err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}
	return 0;
}
