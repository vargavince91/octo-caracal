#include <stdio.h>

__global__ void sum(int a, int b, int *c) {
    *c = a + b;
}

int main (void) {

    int a, b, c;

    int *device_c;
    cudaMalloc((void**) &device_c, sizeof(int));

    printf("This program calculates the sum of two numbers using GPU.\n");

    printf("a=");
    scanf("%d", &a);
    printf("b=");
    scanf("%d", &b);

    sum<<<1,1>>>(a,b,device_c);

    cudaMemcpy(&c, device_c, sizeof(int), cudaMemcpyDeviceToHost);

    printf("%d+%d = %d\n", a, b, c);

    cudaFree(device_c);

    return 0;
}
