// Assignment 4 : GPU Programming
// Author : Bachotti Sai Krishna Shanmukh EE19B009, IIT Madras
// Code for Train Ticket Booking

#include <stdio.h>
#include <cuda.h>

using namespace std;

#define CLASSLIM 25
#define MAXREQ 5000

#define cudaCheck(expr) {\
    cudaError_t __cuda_error = expr;\
    if((__cuda_error) != cudaSuccess)\
    {\
        printf("CUDA error on or before line number %d in file: %s. Error code: %d. Description: %s\n",\
                __LINE__, __FILE__, __cuda_error, cudaGetErrorString(__cuda_error));\
        printf("Terminating execution\n");\
        cudaDeviceReset();\
        exit(0);\
    }\
}


__global__ void batchServer(int* rTCmap, int *rtrainNum, int *rClassNum, int *rSrc, int *rDest, int *rNumSeats, int rCount, int *src, int *maxCap, int N, int *seat, unsigned int *rStatus, unsigned int *ctr){
    __shared__ int rLink[MAXREQ];       // a shared array which links address as current request to value as next request to be processed
    unsigned id = blockDim.x*blockIdx.x + threadIdx.x; // id gen 
    int i;      // local var for iter
    int j;      // local var for iter
    int offset; // local var for offset in seat-matrix update
    int range;  // range of stations in booking in a given request
    int available;  // flag to update if booking can be made
    int trCls;      // a unique number for a given (trainID, ClassID)

    int localCount = 0;    // a thread local int which tracks the number of requests it should process
    int prev_i = 0;            // a thread local int which hold the requestID processed by same thread prior to current requestID
    int rHead = 0;             // a thread local int which holds the first requestID it is supposed to process

    for(i = 0; i < rCount; i++){            // i iterates over requests 
        if(id == rTCmap[i]){                // rTCmap has requestID --> threadID mapping
            if(localCount == 0) {
                rHead = i;                  // smallest requestID mapped to that thread becomes the head
                prev_i = i;                 
            }
            if(prev_i != i){                // updating the rLink 
                rLink[prev_i] = i;          // rLink[ prev request ID] = current request ID
                prev_i = i;
            }
            localCount++;
        }
    }

    // Initialising thread-local i with rHead : smallest requestID mapped to that thread 
    i = rHead;

    
    // Each thread's required portions of rLink array and localCount are ready by this point
    // Iterate over the localcount and process those requests

    for(int k =0; k < localCount; k++){
        if(id == rTCmap[i]){
            // Read from and Update the seat matrix
            // Currently looking at k-th request of a given subset of batch mapped to thread id
            // Need to make sure requested number of seats are available between src and dest (exclusive)
            // i is the request ID in given batch that current thread is processing in it's k-th iteration
            available = 1;
            trCls = rtrainNum[i]*CLASSLIM + rClassNum[i];
            if(rSrc[i] < rDest[i]){
                offset = rSrc[i] - src[rtrainNum[i]];       // local var for offset in seat-matrix update
                range = rDest[i] - rSrc[i];                 // range of stations in booking in a given request
            }

            else {
                offset = src[rtrainNum[i]] - rSrc[i];
                range = rSrc[i] - rDest[i];
            }
            // check for availability in start and transit stations
            for(j = 0; j < range; j++){
                if(seat[trCls*50 + j + offset] + rNumSeats[i] > maxCap[trCls]) {
                    available = 0;
                    break;
                }
            }
            if(available == 1) {            // if availabe : update Seat matrix and counters
                for(j = 0; j < range; j++){
                    seat[trCls*50 + j + offset] += rNumSeats[i];
                }
                rStatus[i] = 1;
                atomicInc(&ctr[0],rCount+1);
                atomicAdd(&ctr[2],rNumSeats[i]*range);
            }
            if(available == 0) atomicInc(&ctr[1],rCount+1);

            // update i with next requestID it should go to
            i = rLink[i];
        }
    }
    
}

int main(int argc,char **argv)
{
    // ==================================== READING TRAINS INFORMATION ==============================
    int N;
    scanf("%d", &N );      //scaning for number of trains

    int i = 0;  // for iterating over trains
    int j =0;   // for iterating over classes in a train

    // Host Arrays
    int *src = (int *) malloc(N * sizeof(int));
    int *dest = (int *) malloc(N * sizeof(int));
    int *classes = (int *) malloc(N * sizeof(int));             // number of classes in a given train
    int *maxCap = (int *) malloc(CLASSLIM * N * sizeof(int));  // a matrix of size N x CLASSLIM to save max capacity of given train,class
    int *TCmap = (int *) malloc(CLASSLIM * N * sizeof(int));  // a matrix to map : (train,class) --> threadId

    memset(maxCap, 0,  CLASSLIM*N*sizeof(int));  // initialize maxCap to 0s

    // Capturing TRAIN, CLASS, SRC, DEST, MAX CAP per class for all trains
    for(i=0; i < N; i++){
        scanf("%d %d %d %d",&i , &classes[i], &src[i], &dest[i]);
        for(j=0; j < classes[i]; j++){
            scanf("%d %d", &j, &maxCap[i*CLASSLIM + j]);
        }
    }

    // Device arrays
    int *dsrc, *dmaxCap, *dTCmap;
    cudaMalloc(&dsrc, N*sizeof(int));
    //cudaCheck(cudaMalloc(&ddest, N*sizeof(int)));
    //cudaCheck(cudaMalloc(&dclasses, N*sizeof(int)));
    cudaMalloc(&dmaxCap, CLASSLIM * N *sizeof(int));
    cudaMalloc(&dTCmap, CLASSLIM * N * sizeof(int));     // map is processed and copied to device
    

    // Copy host to device
    cudaMemcpy(dsrc, src, N*sizeof(int), cudaMemcpyHostToDevice);
    //cudaMemcpy(ddest, dest, N*sizeof(int), cudaMemcpyHostToDevice);
    //cudaMemcpy(dclasses, classes, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dmaxCap, maxCap, CLASSLIM * N * sizeof(int), cudaMemcpyHostToDevice);

    // ===========================================  BATCH PRE PROCESSING ================================================
    int B;
    scanf("%d", &B);      //scaning for number of batches of booking

    // Host var/arrays for Batch processing

    int reqCount = 0;
    int *reqtrainNum = (int *) malloc(MAXREQ*sizeof(int));  // request info in a batch 
    int *reqClassNum = (int *) malloc(MAXREQ*sizeof(int));
    int *reqSrc   = (int *) malloc(MAXREQ*sizeof(int));     // stored in following arrays
    int *reqDest  = (int *) malloc(MAXREQ*sizeof(int));     // trainNum, src, dest and NumSeats
    int *reqNumSeats = (int *) malloc(MAXREQ*sizeof(int));

    int *reqTCmap = (int *) malloc(MAXREQ * sizeof(int));      // a smaller array to map (tr, class) --> threadId in a given batch

    // Device array inputs for Batch processing

    int *dreqtrainNum, *dreqClassNum, *dreqSrc, *dreqDest, *dreqNumSeats, *dreqTCmap;
    cudaMalloc(&dreqtrainNum, MAXREQ*sizeof(int));
    cudaMalloc(&dreqClassNum, MAXREQ*sizeof(int));
    cudaMalloc(&dreqSrc, MAXREQ*sizeof(int));
    cudaMalloc(&dreqDest, MAXREQ*sizeof(int));
    cudaMalloc(&dreqNumSeats, MAXREQ*sizeof(int));
    cudaMalloc(&dreqTCmap, MAXREQ*sizeof(int));

    // Seat Matrix used in Batch processing

    int *seatMatrix;
    cudaMalloc(&seatMatrix, N*CLASSLIM*50*sizeof(int));  // #trains x #max class per train x max #stations
    cudaMemset(seatMatrix,0,N*CLASSLIM*50*sizeof(int));

    // Outputs of Batch Processing
    int *reqStatus = (int *) malloc(MAXREQ*sizeof(int));
    unsigned int *dreqStatus;
    cudaMalloc(&dreqStatus, MAXREQ*sizeof(int));

    int *result = (int *) malloc(3*sizeof(int));
    unsigned int *dresult;
    cudaMalloc(&dresult,3*sizeof(int));

    // gdim declaration
    int gdim;

    // =================================== Reading, Launching and Iterating over batches ================================
    for(i=0; i < B; i++){
        scanf("%d", &reqCount); 

        memset(TCmap, -1,  CLASSLIM * N * sizeof(int));             // setting TCmap to -1 on cpu
        memset(reqTCmap, -1, MAXREQ * sizeof(int));                 // setting reqTCmap to -1 

        // Reading In requests in batch i
        for (j = 0; j < reqCount; j++){
            scanf("%d %d %d %d %d %d", &j, &reqtrainNum[j], &reqClassNum[j], &reqSrc[j], &reqDest[j], &reqNumSeats[j]);
            // Train-Class to request ID mapping
            if(TCmap[reqtrainNum[j]*CLASSLIM + reqClassNum[j]] == -1) TCmap[reqtrainNum[j]*CLASSLIM + reqClassNum[j]] = j;

            reqTCmap[j] = TCmap[reqtrainNum[j]*CLASSLIM + reqClassNum[j]];
        }
        
        // Copying from host to device and setting device arrays

        cudaMemset(dreqStatus,0,MAXREQ*sizeof(int));
        cudaMemset(dresult, 0, 3*sizeof(int));
        cudaMemcpy(dreqtrainNum, reqtrainNum, MAXREQ*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dreqClassNum, reqClassNum, MAXREQ*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dreqSrc, reqSrc, MAXREQ*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dreqDest,reqDest, MAXREQ*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dreqNumSeats, reqNumSeats, MAXREQ*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dreqTCmap, reqTCmap, MAXREQ*sizeof(int), cudaMemcpyHostToDevice);

        // kernel launch :  #threads = #requests 

        gdim = ceil(float(reqCount)/1024);
        batchServer<<<gdim,1024>>>(dreqTCmap, dreqtrainNum,dreqClassNum,dreqSrc,dreqDest,dreqNumSeats, reqCount, dsrc, dmaxCap, N, seatMatrix, dreqStatus,dresult);

        cudaMemcpy(reqStatus,dreqStatus, MAXREQ*sizeof(int), cudaMemcpyDeviceToHost);   // status of each request in batch
        cudaMemcpy(result, dresult, 3*sizeof(int), cudaMemcpyDeviceToHost);         // success, failure and num of tickets
        
        // Printing 

        for(j = 0; j < reqCount; j++){
            if(reqStatus[j] == 1) printf("success\n");
            else printf("failure\n");
        }


        printf("%d %d\n",result[0],result[1]);
        printf("%d\n",result[2]);
    }
    // ====================================================== END of BATCH PROCESSING ================================================

    free(result);
    free(reqStatus);
    free(reqtrainNum);
    free(reqClassNum);
    free(reqSrc);
    free(reqDest);
    free(reqNumSeats);
    free(src);
    free(dest);
    free(classes);
    free(maxCap);
    free(TCmap);
    free(reqTCmap);

    cudaFree(dreqtrainNum);
    cudaFree(dreqClassNum);
    cudaFree(dreqSrc);
    cudaFree(dreqDest);
    cudaFree(dreqNumSeats);
    cudaFree(dreqStatus);
    cudaFree(dreqTCmap);
    cudaFree(dsrc);
    cudaFree(dmaxCap);
    cudaFree(dTCmap);
    cudaFree(seatMatrix);
    cudaFree(dresult);    
}
