#include <stdio.h>
#include <stdlib.h>

#define SIZE 1024

/* $ gcc 02-VectorAdd.c -std=c11 */

/* Source:
 * Channel - NVIDIADeveloper - https://www.youtube.com/user/NVIDIADeveloper
 * Video - CUDACast #2 - Your First CUDA C Program - https://youtu.be/Ed_h2km0liI
 */

/* CPU only code */

void VectorAdd(int *a, int *b, int *c, int n) {
    int i;
    for (i=0; i<n; i++) {
        c[i] = a[i] + b[i];
    }
}

int main() {

    // declare pointers
    int *a, *b, *c;

    // allocate space
    a = (int *)malloc(SIZE*sizeof(int));
    b = (int *)malloc(SIZE*sizeof(int));
    c = (int *)malloc(SIZE*sizeof(int));

    // initialize data
    for (int i=0; i<SIZE; i++) {
        a[i]=i;
        b[i]=i;
        c[i]=0;
    }

    // call VectorAdd function
    VectorAdd(a, b, c, SIZE);

    // show first ten elements so we can check our work
    for (int i=0; i<10; i++) {
        printf("c[%d] = %d\n", i, c[i]);
    }

    // free the memory that we have allocated
    free(a);
    free(b);
    free(c);

    return 0;
}