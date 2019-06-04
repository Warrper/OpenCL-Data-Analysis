//add vector A to vector B in parallel
kernel void add(global const float* A, global const float* B, global float* C) {
	int id = get_global_id(0);
	C[id] = A[id] + B[id];
}

//find lowest value of vector A and vector B in parallel
kernel void getLow(global const float* A, global const float* B, global float* C) { 
	int id = get_global_id(0);
	if (A[id] < B[id]){ 
		C[id] = A[id];
	}
	else { 
		C[id] = B[id];
	}
}

//find highest value of vector A and vector B in parallel
kernel void getHigh(global const float* A, global const float* B, global float* C) { 
	int id = get_global_id(0);
	if (A[id] > B[id]){ 
		C[id] = A[id];
	}
	else { 
		C[id] = B[id];
	}
}

//subtracts B from A and squares the result
kernel void getStdDev(global const float* A, global const float* B, global float* C) { 
	int id = get_global_id(0);
	float temp;
	temp = A[id] - B[id];
	C[id] = temp * temp;
}
