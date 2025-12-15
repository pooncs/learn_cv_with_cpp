#include "npp_resize.h"
#include <npp.h>
#include <cuda_runtime.h>
#include <iostream>

// TODO: Macros for Error Checking

void runNppResizeDemo() {
    int srcW = 512;
    int srcH = 512;
    int dstW = 256;
    int dstH = 256;

    // TODO: Allocate Device Memory using nppiMalloc
    // ...

    // TODO: Copy H -> D
    // ...

    // TODO: Call nppiResize
    // ...

    // TODO: Copy D -> H
    // ...

    // TODO: Verify
    // ...

    // TODO: Free
}
