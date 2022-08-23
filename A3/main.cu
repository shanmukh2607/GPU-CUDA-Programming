#include <stdio.h>
#include <cuda.h>

using namespace std;

// GPU Kernel Definition

__global__ void schedule(int m, int n, int *eT, int *Pr, int *result){
    // a shared int array which has the core id on which a given priority task works
    __shared__ int coreMap[1000];
    // initialization to -1
    coreMap[threadIdx.x] = -1;

    // a counter to keep track of start time of last executed task in the Gantt chart
    __shared__ int ctr,off;
    if (threadIdx.x == 0) ctr = 0;

    // an arr var that keeps track of time until core gets free
    __shared__ int coreCtr[1000];
    coreCtr[threadIdx.x] = 0;

    __shared__ int coreStart[1000];
    coreStart[threadIdx.x] = 0;

    __shared__ int minID[1000];
    minID[threadIdx.x] = threadIdx.x;

    __syncthreads();

    for(int i = 0; i< n; i++){
        if(coreMap[Pr[i]] == -1){
            // find the core which is free and has least id
            // assign that core to the given priority
            if(coreCtr[threadIdx.x] > ctr) coreStart[threadIdx.x] = coreCtr[threadIdx.x];
            else coreStart[threadIdx.x] = ctr;
            minID[threadIdx.x] = threadIdx.x;
            if(threadIdx.x == 0) coreStart[m] = coreStart[m-1];
            __syncthreads();
            // reduction
            for(off = (m+1)/2; off; off /= 2){
                if(threadIdx.x < off){
                    if(coreStart[threadIdx.x] > coreStart[threadIdx.x + off]){
                        coreStart[threadIdx.x] = coreStart[threadIdx.x + off];
                        minID[threadIdx.x] = minID[threadIdx.x + off];
                    }
                    if(coreStart[threadIdx.x] == coreStart[threadIdx.x + off]){
                        if(minID[threadIdx.x] > minID[threadIdx.x + off]) minID[threadIdx.x] = minID[threadIdx.x + off];
                    }     
                }
                if((threadIdx.x ==0) && (off%2==1) && (off!=1)){
                    off++;
                }
                __syncthreads();
                /*
                // searching for min in corner elements in odd length sub arrays
                if(threadIdx.x == 0){
                    if(prevoff == 2*off + 1){
                        if(coreStart[prevoff-1] < coreStart[0]){
                            coreStart[0] = coreStart[prevoff-1];
                            minID[0] = minID[prevoff -1];
                        } 
                        if(coreStart[prevoff-1] == coreStart[0]){
                            if(minID[0] > minID[prevoff-1]) minID[0] = minID[prevoff-1];
                        }
                    }
                    prevoff = off;
                }
                */

            }
            // last element odd 
            //if((m%2==1) && (threadIdx.x==0)){
            //    if(coreStart[m-1] < coreStart[0] ) minID[0] = m-1;
            //}
            
            
            
            if(threadIdx.x == 0) coreMap[Pr[i]] = minID[0];
        }
        __syncthreads();


        if(threadIdx.x == coreMap[Pr[i]]){  // idx is the id of core being used for the task
            //ctr = max(coreCtr[threadIdx.x],ctr);
            if(coreCtr[threadIdx.x] > ctr){
                ctr = coreCtr[threadIdx.x];
            }
            coreCtr[threadIdx.x] = ctr + eT[i];
            result[i] = coreCtr[threadIdx.x];
            //printf("Yeah for task %d the time is %d\n",i,result[i]);
        }
        __syncthreads();
    }
    // addn code to print pr to core map
    //printf("Priority %d is mapped to core %d\n",threadIdx.x, coreMap[threadIdx.x]);
    if(threadIdx.x == 1){
        for(int k = 0; k <m; k++) printf("Priority %d is mapped to core %d\n",k,coreMap[k]);
    }
}

//Complete the following function
void operations ( int m, int n, int *executionTime, int *priority, int *result )  {
    int *d_executionTime, *d_priority, *d_result;
    cudaMalloc(&d_executionTime, n*sizeof(int));
    cudaMalloc(&d_priority, n*sizeof(int));
    cudaMalloc(&d_result, n*sizeof(int));
    cudaMemcpy(d_executionTime, executionTime, n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_priority, priority, n*sizeof(int), cudaMemcpyHostToDevice);
    schedule<<<1,m>>>(m,n,d_executionTime,d_priority,d_result);
    cudaMemcpy(result, d_result, n*sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_executionTime);
    cudaFree(d_priority);
    cudaFree(d_result);
}

int main(int argc,char **argv)
{
    int m,n;
    //Input file pointer declaration
    FILE *inputfilepointer;
    
    //File Opening for read
    char *inputfilename = argv[1];
    inputfilepointer    = fopen( inputfilename , "r");
    
    //Checking if file ptr is NULL
    if ( inputfilepointer == NULL )  {
        printf( "input.txt file failed to open." );
        return 0; 
    }

    fscanf( inputfilepointer, "%d", &m );      //scaning for number of cores
    fscanf( inputfilepointer, "%d", &n );      //scaning for number of tasks
   
   //Taking execution time and priorities as input	
    int *executionTime = (int *) malloc ( n * sizeof (int) );
    int *priority = (int *) malloc ( n * sizeof (int) );
    for ( int i=0; i< n; i++ )  {
            fscanf( inputfilepointer, "%d", &executionTime[i] );
    }

    for ( int i=0; i< n; i++ )  {
            fscanf( inputfilepointer, "%d", &priority[i] );
    }

    //Allocate memory for final result output 
    int *result = (int *) malloc ( (n) * sizeof (int) );
    for ( int i=0; i<n; i++ )  {
        result[i] = 0;
    }
    
     cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;
    cudaEventRecord(start,0);

    //==========================================================================================================
	

	operations ( m, n, executionTime, priority, result ); 
	
    //===========================================================================================================
    
    
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time taken by function to execute is: %.6f ms\n", milliseconds);
    
    // Output file pointer declaration
    char *outputfilename = argv[2]; 
    FILE *outputfilepointer;
    outputfilepointer = fopen(outputfilename,"w");

    //Total time of each task: Final Result
    for ( int i=0; i<n; i++ )  {
        fprintf( outputfilepointer, "%d ", result[i]);
    }

    fclose( outputfilepointer );
    fclose( inputfilepointer );
    
    free(executionTime);
    free(priority);
    free(result);
    
    
}
/*
                // searching for min in corner elements in odd length sub arrays
                if(threadIdx.x == 0){
                    if(prevoff == 2*off + 1){
                        if(coreStart[prevoff-1] < coreStart[0]){
                            coreStart[0] = coreStart[prevoff-1];
                            minID[0] = minID[prevoff -1];
                        } 
                        if(coreStart[prevoff-1] == coreStart[0]){
                            if(minID[0] > minID[prevoff-1]) minID[0] = minID[prevoff-1];
                        }
                    }
                    prevoff = off;
                }
                */