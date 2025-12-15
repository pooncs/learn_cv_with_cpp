#include "npp_resize.h"
#include <npp.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(err); \
        } \
    } while (0)

#define NPP_CHECK(call) \
    do { \
        NppStatus status = call; \
        if (status != NPP_SUCCESS) { \
            std::cerr << "NPP Error: " << status << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(status); \
        } \
    } while (0)

void runNppResizeDemo() {
    int srcW = 512;
    int srcH = 512;
    int dstW = 256;
    int dstH = 256;

    std::cout << "NPP Resize Demo: " << srcW << "x" << srcH << " -> " << dstW << "x" << dstH << std::endl;

    // 1. Host Data (Gradient)
    std::vector<Npp8u> h_src(srcW * srcH);
    for (int y = 0; y < srcH; ++y) {
        for (int x = 0; x < srcW; ++x) {
            h_src[y * srcW + x] = (x + y) % 256;
        }
    }

    // 2. Allocate Device Memory (using NPP allocator for padding/alignment)
    Npp8u *d_src = nullptr;
    int srcStep = 0;
    d_src = nppiMalloc_8u_C1(srcW, srcH, &srcStep);

    Npp8u *d_dst = nullptr;
    int dstStep = 0;
    d_dst = nppiMalloc_8u_C1(dstW, dstH, &dstStep);

    // 3. Copy H -> D
    CUDA_CHECK(cudaMemcpy2D(d_src, srcStep, h_src.data(), srcW, srcW, srcH, cudaMemcpyHostToDevice));

    // 4. Resize
    NppiSize srcSize = {srcW, srcH};
    NppiRect srcROI = {0, 0, srcW, srcH};
    NppiSize dstSize = {dstW, dstH};
    NppiRect dstROI = {0, 0, dstW, dstH};

    // Note: nppiResize_8u_C1R is deprecated in newer CUDA versions in favor of SqrPixel variants or others,
    // but often still available. If not, we use nppiResizeSqrPixel_8u_C1R.
    // Let's use standard resize if available, or SqrPixel.
    // Checking nppi_geometry_transforms.h... nppiResize_8u_C1R takes x/y factors.
    
    double xFactor = (double)dstW / srcW;
    double yFactor = (double)dstH / srcH;
    
    NPP_CHECK(nppiResize_8u_C1R(d_src, srcSize, srcStep, srcROI,
                                d_dst, dstStep, dstSize, dstROI,
                                NPPI_INTER_LINEAR));

    // 5. Copy D -> H
    std::vector<Npp8u> h_dst(dstW * dstH);
    CUDA_CHECK(cudaMemcpy2D(h_dst.data(), dstW, d_dst, dstStep, dstW, dstH, cudaMemcpyDeviceToHost));

    // 6. Verify (Check center pixel)
    // Center of src (256, 256) should map roughly to center of dst (128, 128)
    int srcVal = h_src[256 * srcW + 256];
    int dstVal = h_dst[128 * dstW + 128];
    
    std::cout << "Src(256,256) = " << srcVal << std::endl;
    std::cout << "Dst(128,128) = " << dstVal << std::endl;

    // Allow some interpolation error
    if (abs(srcVal - dstVal) < 5) {
        std::cout << "Result: PASS" << std::endl;
    } else {
        std::cout << "Result: FAIL" << std::endl;
    }

    nppiFree(d_src);
    nppiFree(d_dst);
}
