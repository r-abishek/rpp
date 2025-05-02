#include "rpp.h"
#include "handle.hpp"
#include <math.h>
#include <algorithm>
#include <omp.h>
#include "rpp_cpu_simd_load_store.hpp"
#include "rpp_cpu_simd_math.hpp"


// -------------------- fisheye --------------------

template <typename T>
RppStatus fisheye_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr, RppiROI *roiPoints,
                             Rpp32u nbatchSize,
                             RppiChnFormat chnFormat, Rpp32u channel, rpp::Handle& handle);

// -------------------- snow --------------------

template <typename T>
RppStatus snow_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
                          Rpp32f *batch_strength,
                          RppiROI *roiPoints, Rpp32u nbatchSize,
                          RppiChnFormat chnFormat, Rpp32u channel, rpp::Handle& handle);