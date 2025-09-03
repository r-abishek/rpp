#include "hip_tensor_executors.hpp"

// Device-side bin conversion: clamp to [0,255]
__device__ __forceinline__ int to_bin_0_255(unsigned char x) { return x; }

__global__ void build_lut_from_hist_kernel(const unsigned int* __restrict__ hist,
                                           unsigned char* __restrict__ lut,
                                           const int* __restrict__ img_sizes,
                                           int batchSize)
{
    int batch = blockIdx.x;
    if (batch >= batchSize) return;

    int tid = threadIdx.x;
    __shared__ unsigned int cdf_shared[256];
    __shared__ unsigned int min_cdf_shared;
    if (tid == 0) min_cdf_shared = 0;
    __syncthreads();

    // Compute CDF and min_cdf
    unsigned int cdf = 0;
    for (int i = 0; i < 256; ++i)
    {
        if (tid == 0) cdf_shared[i] = 0;
    }
    __syncthreads();

    for (int i = tid; i < 256; i += blockDim.x)
    {
        unsigned int val = hist[batch * 256 + i];
        atomicAdd(&cdf_shared[i], val);
    }
    __syncthreads();

    if (tid == 0)
    {
        unsigned int cdf_accum = 0;
        for (int i = 0; i < 256; ++i)
        {
            cdf_accum += cdf_shared[i];
            cdf_shared[i] = cdf_accum;
            if (min_cdf_shared == 0 && cdf_shared[i] != 0)
                min_cdf_shared = cdf_shared[i];
        }
    }
    __syncthreads();

    int N = img_sizes[batch];
    for (int i = tid; i < 256; i += blockDim.x)
    {
        lut[batch * 256 + i] = (unsigned char)(roundf((float)(cdf_shared[i] - min_cdf_shared) * 255.0f / (N - min_cdf_shared + 1e-5f)));
    }
}

// Kernel to collect histogram for pln1 (single-channel, planar), batch-capable
__global__ void collect_hist_pln_hip_tensor_batch(const unsigned char *__restrict__ srcPtr,
                                                  RpptROIPtr roiTensorPtrSrc,
                                                   uint3 srcStridesNCH,
                                                   unsigned int *__restrict__ hist)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
        return;

    uint srcIdx = (id_z * srcStridesNCH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    uint histOffset = id_z * 256;
    uint8_t pixVal = srcPtr[srcIdx];
    atomicAdd(&hist[histOffset + pixVal], 1);
}

__global__ void apply_lut_pln1_hip_tensor(const unsigned char *__restrict__ srcPtr,
                                          uint3 srcStridesNCH,
                                          unsigned char *__restrict__ dstPtr,
                                          uint3 dstStridesNCH,
                                          const unsigned char* __restrict__ lut,
                                          RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
        return;

    uint srcIdx = (id_z * srcStridesNCH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    unsigned char pixVal = srcPtr[srcIdx];
    dstPtr[dstIdx] = lut[pixVal];
}

RppStatus hip_exec_histogram_equalize_tensor(Rpp8u *srcPtr,
                                             RpptDescPtr srcDescPtr,
                                             Rpp8u *dstPtr,
                                             RpptDescPtr dstDescPtr,
                                             RpptROIPtr roiTensorPtrSrc,
                                             RpptRoiType roiType,
                                             rpp::Handle& handle)
{
    if (roiType == RpptRoiType::LTRB)
        hip_exec_roi_converison_ltrb_to_xywh(roiTensorPtrSrc, handle);

    int batchSize = dstDescPtr->n;
    const int hist_size = 256;

    // Use handle's scratch buffers for host and device
    unsigned int* hist = reinterpret_cast<unsigned int*>(handle.GetInitHandle()->mem.mcpu.scratchBufferHost);
    unsigned char* lut = reinterpret_cast<unsigned char*>(hist + batchSize * hist_size);
    unsigned int* d_hist = reinterpret_cast<unsigned int*>(handle.GetInitHandle()->mem.mgpu.scratchBufferHip.floatmem);
    unsigned char* d_lut = reinterpret_cast<unsigned char*>(d_hist + batchSize * hist_size);

    // 1. Zero device histogram
    hipMemsetAsync(d_hist, 0, batchSize * hist_size * sizeof(unsigned int), handle.GetStream());

    int globalThreads_x = srcDescPtr->w;
    int globalThreads_y = srcDescPtr->h;
    int globalThreads_z = srcDescPtr->n;
    // 2. Collect histogram for all batches in one kernel launch
    hipLaunchKernelGGL(collect_hist_pln_hip_tensor_batch,
                       dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                       dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       roiTensorPtrSrc,
                       make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                       d_hist);

    // 3. Build LUTs on device for each batch
    // Prepare image sizes array on host and copy to device
    std::vector<int> img_sizes(batchSize);
    for (int b = 0; b < batchSize; ++b)
    {
        int w = roiTensorPtrSrc[b].xywhROI.roiWidth;
        int h = roiTensorPtrSrc[b].xywhROI.roiHeight;
        img_sizes[b] = w * h;
    }
    int* d_img_sizes;
    hipMalloc(&d_img_sizes, batchSize * sizeof(int));
    hipMemcpyAsync(d_img_sizes, img_sizes.data(), batchSize * sizeof(int), hipMemcpyHostToDevice, handle.GetStream());

    hipLaunchKernelGGL(build_lut_from_hist_kernel, dim3(batchSize), dim3(256), 0, handle.GetStream(),
        d_hist, d_lut, d_img_sizes, batchSize
    );
    hipFree(d_img_sizes);

    globalThreads_x = dstDescPtr->w;
    globalThreads_y = dstDescPtr->h;
    globalThreads_z = dstDescPtr->n;

    hipLaunchKernelGGL(apply_lut_pln1_hip_tensor,
                       dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                       dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                       dstPtr,
                       make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                       d_lut,
                       roiTensorPtrSrc);

    return RPP_SUCCESS;
}
