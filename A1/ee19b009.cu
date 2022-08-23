#include <cuda.h>
#include <bits/stdc++.h>
#include <fstream>
using namespace std;

ofstream outfile; //the handle for printing the output

// complete the following kernel...
__global__ void per_row_column_kernel(long int *A, long int *B, long int *C,long int m, long int n){

	unsigned ii = blockIdx.x * blockDim.x + threadIdx.x;
	if(ii < m){
		for(unsigned jj = 0; jj < n; jj++){
			C[ii*n + jj] = (A[ii*n + jj] + B[ii + m*jj])*(B[ii + m*jj] - A[ii*n + jj]);
		}
	}
    // jj is loop running index
    // a_ij = A[ii*n + jj] runs row-wise
    // bt_ij = B[ii + m*jj] runs col-wise
	// cij  runs row-wise
}

// complete the following kernel...
__global__ void per_column_row_kernel(long int *A, long int *B, long int *C,long int m, long int n){

    unsigned jj = blockIdx.x*blockDim.x*blockDim.y + threadIdx.x * blockDim.x + threadIdx.y;
    if (jj < n){
		for(unsigned ii=0; ii < m; ii++){
			C[jj + n*ii] = (A[jj + n*ii] + B[jj*m + ii])*(B[jj*m + ii] - A[jj + n*ii]);
		}
    }
        // ii is loop running index
    	// a_ij = A[jj + n*ii] runs col-wise
		// bt_ij = B[jj*m + ii] runs row-wise
}

// complete the following kernel...
__global__ void per_element_kernel(long int *A, long int *B, long int *C,long int m, long int n){

	unsigned ii = blockIdx.y * blockDim.x + threadIdx.x;
	unsigned jj = blockIdx.x * blockDim.y + threadIdx.y;

	if ((ii < m) && (jj < n)){
		C[n*ii + jj] = (A[n*ii + jj] + B[ii + m*jj])*(B[ii + m*jj] - A[n*ii + jj]);
	}
	// Each thread computes an element
} 

/**
 * Prints any 1D array in the form of a matrix 
 * */
void printMatrix(long int *arr, long int rows, long int cols, char* filename) {

	outfile.open(filename);
	for(long int i = 0; i < rows; i++) {
		for(long int j = 0; j < cols; j++) {
			outfile<<arr[i * cols + j]<<" ";
		}
		outfile<<"\n";
	}
	outfile.close();
}

int main(int argc,char **argv){

	//variable declarations
	long int m,n;	
	cin>>m>>n;	

	//host_arrays 
	long int *h_a,*h_b,*h_c;

	//device arrays 
	long int *d_a,*d_b,*d_c;
	
	//Allocating space for the host_arrays 
	h_a = (long int *) malloc(m * n * sizeof(long int));
	h_b = (long int *) malloc(m * n * sizeof(long int));	
	h_c = (long int *) malloc(m * n * sizeof(long int));	

	//Allocating memory for the device arrays 
	cudaMalloc(&d_a, m * n * sizeof(long int));
	cudaMalloc(&d_b, m * n * sizeof(long int));
	cudaMalloc(&d_c, m * n * sizeof(long int));

	//Read the input matrix A 
	for(long int i = 0; i < m * n; i++) {
		cin>>h_a[i];
	}

	//Read the input matrix B 
	for(long int i = 0; i < m * n; i++) {
		cin>>h_b[i];
	}

	//Transfer the input host arrays to the device 
	cudaMemcpy(d_a, h_a, m * n * sizeof(long int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, m * n * sizeof(long int), cudaMemcpyHostToDevice);

	long int gridDimx, gridDimy;
	//Launch the kernels 
	/**
	 * Kernel 1 - per_row_column_kernel
	 * To be launched with 1D grid, 1D block
	 * */
	gridDimx = ceil(float(m) / 1024);
	dim3 grid1(gridDimx,1,1);
	dim3 block1(1024,1,1);
	per_row_column_kernel<<<grid1,block1>>>(d_a,d_b,d_c,m,n);
	cudaDeviceSynchronize();
	cudaMemcpy(h_c, d_c, m * n * sizeof(long int), cudaMemcpyDeviceToHost);
	printMatrix(h_c, m, n,"kernel1.txt");
	
	/**
	 * Kernel 2 - per_column_row_kernel
	 * To be launched with 1D grid, 2D block
	 * */ 
	
	gridDimx = ceil(float(n) / 1024);
	dim3 grid2(gridDimx,1,1);
	dim3 block2(32,32,1);
	per_column_row_kernel<<<grid2,block2>>>(d_a,d_b,d_c,m,n);
	cudaDeviceSynchronize();
	cudaMemcpy(h_c, d_c, m * n * sizeof(long int), cudaMemcpyDeviceToHost);
	printMatrix(h_c, m, n,"kernel2.txt");

	/**
	 * Kernel 3 - per_element_kernel
	 * To be launched with 2D grid, 2D block
	 * */
	
	gridDimx = ceil(float(n) / 16);
	gridDimy = ceil(float(m) / 64);
	dim3 grid3(gridDimx,gridDimy,1);
	dim3 block3(64,16,1);
	per_element_kernel<<<grid3,block3>>>(d_a,d_b,d_c,m,n);
	cudaDeviceSynchronize();
	cudaMemcpy(h_c, d_c, m * n * sizeof(long int), cudaMemcpyDeviceToHost);
	printMatrix(h_c, m, n,"kernel3.txt");
    

	return 0;
}
