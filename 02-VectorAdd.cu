#include <stdio.h>
#include <stdlib.h>

#define SIZE 1024

/* $ nvcc 02-VectorAdd.cu -o 02-VectorAdd.out */

/* Source:
 * Channel - NVIDIADeveloper - https://www.youtube.com/user/NVIDIADeveloper
 * Video - CUDACast #2 - Your First CUDA C Program - https://youtu.be/Ed_h2km0liI
 */

/* GPU Code */
// 1. paralellize function
// 2/a. allocate memory on the GPU
// 2/b. move our data over
// 3. modify function call in order to enable to launch on the GPU





// __global__ // it tells the compiler that this function is going to be executed on the GPU and it's callable from the host
__global__ void VectorAdd(int *a, int *b, int *c, int n) {
    // a way for each thread to identify itself
    // threadIdx // read-only
    int i = threadIdx.x;
    if (i<n) {
        c[i] = a[i] + b[i];
    }
}

int main() {

    // declare pointers
    int *a, *b, *c;
    int *device_a, *device_b, *device_c;

    // allocate memory on CPU
    a = (int *)malloc(SIZE*sizeof(int));
    b = (int *)malloc(SIZE*sizeof(int));
    c = (int *)malloc(SIZE*sizeof(int));

    // initialize data
    for (int i=0; i<SIZE; i++) {
        a[i]=i;
        b[i]=i;
        c[i]=0;
    }

    // allocate memory on GPU
    cudaMalloc(&device_a, SIZE*sizeof(int));
    cudaMalloc(&device_b, SIZE*sizeof(int));
    cudaMalloc(&device_c, SIZE*sizeof(int));

    // copy initialized data to the GPU
    // cudaMemcpy(destination, source, nr of bytes, direction)
    cudaMemcpy(device_a, a, SIZE*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, b, SIZE*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_c, c, SIZE*sizeof(int), cudaMemcpyHostToDevice);

    // call function
    // specify the launch configuration of this kernel
    // <<<BLOCKS, THREADS in BLOCK>>>
    VectorAdd<<<1, SIZE>>>(device_a, device_b, device_c, SIZE);

    // copy back the result
    cudaMemcpy(a, device_a, SIZE*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(b, device_b, SIZE*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(c, device_c, SIZE*sizeof(int), cudaMemcpyDeviceToHost);

    // show first ten elements so we can check our work
    for (int i=0; i<10; i++) {
        printf("c[%d] = %d\n", i, c[i]);
    }

    // free the memory that we have allocated on CPU
    free(a);
    free(b);
    free(c);

    // free device side memory as well
    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);

    return 0;
}