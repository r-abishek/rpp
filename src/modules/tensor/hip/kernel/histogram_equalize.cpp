#include "hip_tensor_executors.hpp"

__device__ const float4 yR_f4 = (float4)0.299f;
__device__ const float4 yG_f4 = (float4)0.587f;
__device__ const float4 yB_f4 = (float4)0.114f;

__device__ const float4 cbR_f4 = (float4)-0.168736f;
__device__ const float4 cbG_f4 = (float4)-0.331264f;
__device__ const float4 cbB_f4 = (float4)0.5f;

__device__ const float4 crR_f4 = (float4)0.5f;
__device__ const float4 crG_f4 = (float4)-0.418688f;
__device__ const float4 crB_f4 = (float4)-0.081312f;

__device__ const float4 maxVal255_f4 = (float4)255.0f;
__device__ const float4 maxVal128_f4 = (float4)128.0f;

__device__ inline float4 clamp(float4 v, float lo, float hi)
{
    v.x = fminf(fmaxf(v.x, lo), hi);
    v.y = fminf(fmaxf(v.y, lo), hi);
    v.z = fminf(fmaxf(v.z, lo), hi);
    v.w = fminf(fmaxf(v.w, lo), hi);

    return v;
}

__device__ inline void ycbcr_to_rgb_hip_compute(d_float24 &rgb_f24, d_float8 &y_f8, d_float8 &cb_f8, d_float8 &cr_f8)
{
    // Subtract 128 from Cb and Cr
    cb_f8.f4[0] -= (float4)128.0f;
    cb_f8.f4[1] -= (float4)128.0f;
    cr_f8.f4[0] -= (float4)128.0f;
    cr_f8.f4[1] -= (float4)128.0f;

    // R = Y + 1.402 * Cr
    rgb_f24.f4[0] = clamp((y_f8.f4[0] + (float4)1.402f * cr_f8.f4[0]), 0.0f, 255.0f);
    rgb_f24.f4[1] = clamp((y_f8.f4[1] + (float4)1.402f * cr_f8.f4[1]), 0.0f, 255.0f);

    // G = Y - 0.344136 * Cb - 0.714136 * Cr
    rgb_f24.f4[2] = clamp((y_f8.f4[0] - ((float4)0.344136f * cb_f8.f4[0]) - ((float4)0.714136f * cr_f8.f4[0])), 0.0f, 255.0f);
    rgb_f24.f4[3] = clamp((y_f8.f4[1] - ((float4)0.344136f * cb_f8.f4[1]) - ((float4)0.714136f * cr_f8.f4[1])), 0.0f, 255.0f);

    // B = Y + 1.772 * Cb
    rgb_f24.f4[4] = clamp((y_f8.f4[0] + (float4)1.772f * cb_f8.f4[0]), 0.0f, 255.0f);
    rgb_f24.f4[5] = clamp((y_f8.f4[1] + (float4)1.772f * cb_f8.f4[1]), 0.0f, 255.0f);
}

__device__ inline void ycbcr_hip_compute(d_float24 &rgb_f24, d_float8 &y_f8, d_float8 &cb_f8, d_float8 &cr_f8)
{
    // Y
    y_f8.f4[0] = clamp((rgb_f24.f4[0] * yR_f4) +
                        (rgb_f24.f4[2] * yG_f4) +
                        (rgb_f24.f4[4] * yB_f4), 0.0f, 255.0f);

    y_f8.f4[1] = clamp((rgb_f24.f4[1] * yR_f4) +
                        (rgb_f24.f4[3] * yG_f4) +
                        (rgb_f24.f4[5] * yB_f4), 0.0f, 255.0f);

    // Cb
    cb_f8.f4[0] = clamp((rgb_f24.f4[0] * cbR_f4) +
                         (rgb_f24.f4[2] * cbG_f4) +
                         (rgb_f24.f4[4] * cbB_f4) + maxVal128_f4,
                         0.0f, 255.0f);

    cb_f8.f4[1] = clamp((rgb_f24.f4[1] * cbR_f4) +
                         (rgb_f24.f4[3] * cbG_f4) +
                         (rgb_f24.f4[5] * cbB_f4) + maxVal128_f4,
                         0.0f, 255.0f);

    // Cr
    cr_f8.f4[0] = clamp((rgb_f24.f4[0] * crR_f4) +
                         (rgb_f24.f4[2] * crG_f4) +
                         (rgb_f24.f4[4] * crB_f4) + maxVal128_f4,
                         0.0f, 255.0f);

    cr_f8.f4[1] = clamp((rgb_f24.f4[1] * crR_f4) +
                         (rgb_f24.f4[3] * crG_f4) +
                         (rgb_f24.f4[5] * crB_f4) + maxVal128_f4,
                         0.0f, 255.0f);
}

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
    dstPtr[dstIdx] = lut[id_z * 256 + pixVal];
}

__global__ void convert_pkd3_to_yuv(unsigned char *__restrict__ srcPtr,
                                    uint2 srcStridesNH,
                                    unsigned char *__restrict__ yPtr,
                                    unsigned char *__restrict__ cbPtr,
                                    unsigned char *__restrict__ crPtr,
                                    uint2 dstStridesWH,
                                    RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
        return;

    uint srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    uint dstIdx = (id_z * dstStridesWH.y * dstStridesWH.x) + (id_y * dstStridesWH.x) + id_x;
    d_float24 rgb_f24;
    d_float8 y_f8, cb_f8, cr_f8;
    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &rgb_f24);
    ycbcr_hip_compute(rgb_f24, y_f8, cb_f8, cr_f8);
    rpp_hip_pack_float8_and_store8(yPtr + dstIdx, &y_f8);
    rpp_hip_pack_float8_and_store8(cbPtr + dstIdx, &cb_f8);
    rpp_hip_pack_float8_and_store8(crPtr + dstIdx, &cr_f8);
}

// New kernel to rebuild RGB from equalized Y and original Cb/Cr (NHWC)
__global__ void convert_yuv_to_pkd3(unsigned char *__restrict__ yPtr,
                                    unsigned char *__restrict__ cbPtr,
                                    unsigned char *__restrict__ crPtr,
                                    uint2 srcStridesWH,
                                    unsigned char *__restrict__ dstPtr,
                                    uint2 dstStridesNH,
                                    RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
        return;

    uint srcIdx = (id_z * srcStridesWH.y * srcStridesWH.x) + (id_y * srcStridesWH.x) + id_x;
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + (id_x * 3);

    d_float24 rgb_f24;
    d_float8 y_f8, cb_f8, cr_f8;
    rpp_hip_load8_and_unpack_to_float8(yPtr + srcIdx, &y_f8);
    rpp_hip_load8_and_unpack_to_float8(cbPtr + srcIdx, &cb_f8);
    rpp_hip_load8_and_unpack_to_float8(crPtr + srcIdx, &cr_f8);
    ycbcr_to_rgb_hip_compute(rgb_f24, y_f8, cb_f8, cr_f8);
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &rgb_f24);
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

    if(srcDescPtr->c == 3)
    {
        const size_t planeSize = static_cast<size_t>(srcDescPtr->w) * srcDescPtr->h * srcDescPtr->n;
        Rpp8u *yuvBuf; 
        hipMalloc((&yuvBuf), planeSize * 3);
        Rpp8u *yBuf = yuvBuf;
        Rpp8u *cbBuf = yuvBuf + planeSize;
        Rpp8u *crBuf = yuvBuf + (planeSize * 2);
        if(srcDescPtr->layout == RpptLayout::NHWC)
        {
            int globalThreads_x = (srcDescPtr->strides.hStride + 7) >> 3; // each thread does 8 pixels
            int globalThreads_y = srcDescPtr->h;
            int globalThreads_z = srcDescPtr->n;

            hipLaunchKernelGGL(convert_pkd3_to_yuv,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               yBuf,
                               cbBuf,
                               crBuf,
                               make_uint2(srcDescPtr->w, srcDescPtr->h),
                               roiTensorPtrSrc);

            // Use handle's scratch buffers for host and device
            unsigned int* hist = reinterpret_cast<unsigned int*>(handle.GetInitHandle()->mem.mcpu.scratchBufferHost);
            unsigned char* lut = reinterpret_cast<unsigned char*>(hist + batchSize * hist_size);
            unsigned int* d_hist = reinterpret_cast<unsigned int*>(handle.GetInitHandle()->mem.mgpu.scratchBufferHip.floatmem);
            unsigned char* d_lut = reinterpret_cast<unsigned char*>(d_hist + batchSize * hist_size);

            // 1. Zero device histogram
            hipMemsetAsync(d_hist, 0, batchSize * hist_size * sizeof(unsigned int), handle.GetStream());

            globalThreads_x = srcDescPtr->w;
            globalThreads_y = srcDescPtr->h;
            globalThreads_z = srcDescPtr->n;
            // 2. Collect histogram for all batches in one kernel launch
            hipLaunchKernelGGL(collect_hist_pln_hip_tensor_batch,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               yBuf,
                               roiTensorPtrSrc,
                               make_uint3(srcDescPtr->w * srcDescPtr->h, srcDescPtr->w * srcDescPtr->h, srcDescPtr->w),
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
                               yBuf,
                               make_uint3(srcDescPtr->w * srcDescPtr->h, srcDescPtr->w * srcDescPtr->h, srcDescPtr->w),
                               yBuf,
                               make_uint3(srcDescPtr->w * srcDescPtr->h, srcDescPtr->w * srcDescPtr->h, srcDescPtr->w),
                               d_lut,
                               roiTensorPtrSrc);

                globalThreads_x = (dstDescPtr->w + 7) >> 3; // each thread does 8 pixels
                globalThreads_y = dstDescPtr->h;
                globalThreads_z = dstDescPtr->n;
                
                hipLaunchKernelGGL(convert_yuv_to_pkd3,
                                   dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                                   dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                                   0,
                                   handle.GetStream(),
                                   yBuf,
                                   cbBuf,
                                   crBuf,
                                   make_uint2(srcDescPtr->w, srcDescPtr->h),
                                   dstPtr,
                                   make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                                   roiTensorPtrSrc);
        }

        hipFree(yuvBuf);
        return RPP_SUCCESS;
    }

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
