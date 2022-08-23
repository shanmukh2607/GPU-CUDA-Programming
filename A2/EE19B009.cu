#include<iostream>
#include<sys/time.h>
#include<cuda.h>
using namespace std;


// write kernels here...
/* draft 1
__global__ multiply1(int *mul1, int *mul2, int *product int q, int r, int s){
	extern __shared__ int dotrow[r];
	dotrow[threadIdx.x] = mul1[blockIdx.x*r + threadIdx.x] * mul2[blockIdx.y*r + threadIdx.x];
	__syncthreads();
	sum = 0;
	for (int k = 0; k<r; k++){
		sum += dotrow[k]; // adding up sum of elements in dotrow to it's first element
	}
	product[blockIdx.x*s + blockIdx.y] = sum;
}
// multiply1<<<dim2(q,s),r>>>(); 
*/

// draft 2
__global__ multiplyCD(int *matC, int *matD, int *product, int q, int r, int s){
	__shared__ int row[1024];
	for(int k=0; k<r;k++){
		row[threadIdx.x] += matC[blockIdx.x*r + k] * matD[threadIdx.x*r + k]; // mem coalescing is prob
	}
	product[blockIdx.x*s + threadIdx.x] = row[threadIdx.x];
}

__global__ multiplyABY(int *matA, int *matB, int *matY, int *matX, int p, int q, int s){
	__shared__ int col[1024];
	for(int k=0; k<q ; k++){
		col[threadIdx.x] += (matA[blockIdx.x*q + k] + matB[blockIdx.x + k*p] )* matY[threadIdx.x + k*s];
	}
	matX[blockIdx.x*s + threadIdx.x] = col[threadIdx.x];
}



// function to compute the output matrix
void compute(int p, int q, int r, int s, int *h_matrixA, int *h_matrixB, 
	         int *h_matrixC, int *h_matrixD, int *h_matrixX) {
	// variable declarations for device arrays
	int *dmatrixA, *dmatrixB, *dmatrixC, *dmatrixD, *dmatrixX, *dmatrixY;
	// allocate memory...
	cudaMalloc(&dmatrixA, p*q*sizeof(int));
	cudaMalloc(&dmatrixB, q*p*sizeof(int));
	cudaMalloc(&dmatrixC, q*r*sizeof(int));
	cudaMalloc(&dmatrixD, s*r*sizeof(int));
	cudaMalloc(&dmatrixX, p*s*sizeof(int));
	cudaMalloc(&dmatrixY, q*s*sizeof(int));
	// copy the values...
	cudaMemcpy(dmatrixA, h_matrixA, p*q*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dmatrixB, h_matrixB, q*p*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dmatrixC, h_matrixC, q*r*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dmatrixD, h_matrixD, s*r*sizeof(int), cudaMemcpyHostToDevice);
	// call the kernels for doing required computations...
	multiplyCD<<<q,s>>>(dmatrixC,dmatrixD,dmatrixY, q,r,s);
	multiplyABY<<<p,s>>>(dmatrixA,dmatrixB,dmatrixY,dmatrixX,p,q,s);
	// copy the result back...
	cudaMemcpy(h_matrixX, dmatrixX, p*s*sizeof(int), cudaMemcpyDeviceToHost);
	// deallocate the memory...
}

// function to read the input matrices from the input file
void readMatrix(FILE *inputFilePtr, int *matrix, int rows, int cols) {
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			fscanf(inputFilePtr, "%d", &matrix[i*cols+j]);
		}
	}
}

// function to write the output matrix into the output file
void writeMatrix(FILE *outputFilePtr, int *matrix, int rows, int cols) {
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			fprintf(outputFilePtr, "%d ", matrix[i*cols+j]);
		}
		fprintf(outputFilePtr, "\n");
	}
}

int main(int argc, char **argv) {
	// variable declarations
	int p, q, r, s;
	int *matrixA, *matrixB, *matrixC, *matrixD, *matrixX;
	struct timeval t1, t2;
	double seconds, microSeconds;

	// get file names from command line
	char *inputFileName = argv[1];
	char *outputFileName = argv[2];

	// file pointers
	FILE *inputFilePtr, *outputFilePtr;
    
    inputFilePtr = fopen(inputFileName, "r");
	if(inputFilePtr == NULL) {
	    printf("Failed to open the input file.!!\n"); 
		return 0;
	}

	// read input values
	fscanf(inputFilePtr, "%d %d %d %d", &p, &q, &r, &s);

	// allocate memory and read input matrices
	matrixA = (int*) malloc(p * q * sizeof(int));
	matrixB = (int*) malloc(q * p * sizeof(int));
	matrixC = (int*) malloc(q * r * sizeof(int));
	matrixD = (int*) malloc(s * r * sizeof(int));
	readMatrix(inputFilePtr, matrixA, p, q);
	readMatrix(inputFilePtr, matrixB, q, p);
	readMatrix(inputFilePtr, matrixC, q, r);
	readMatrix(inputFilePtr, matrixD, s, r);

	// allocate memory for output matrix
	matrixX = (int*) malloc(p * s * sizeof(int));

	// call compute function to get the output matrix. it is expected that 
	// the compute function will store the result in matrixX.
	gettimeofday(&t1, NULL);
	compute(p, q, r, s, matrixA, matrixB, matrixC, matrixD, matrixX);
	cudaDeviceSynchronize();
	gettimeofday(&t2, NULL);

	// print the time taken by the compute function
	seconds = t2.tv_sec - t1.tv_sec;
	microSeconds = t2.tv_usec - t1.tv_usec;
	printf("Time taken (ms): %.3f\n", 1000*seconds + microSeconds/1000);

	// store the result into the output file
	outputFilePtr = fopen(outputFileName, "w");
	writeMatrix(outputFilePtr, matrixX, p, s);

	// close files
    fclose(inputFilePtr);
    fclose(outputFilePtr);

	// deallocate memory
	free(matrixA);
	free(matrixB);
	free(matrixC);
	free(matrixD);
	free(matrixX);

	return 0;
}