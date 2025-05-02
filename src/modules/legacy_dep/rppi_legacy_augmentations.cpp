#include "rppi_validate.hpp"
#include "rppi_legacy_augmentations.h"
#include "host_legacy_executors.hpp"

#ifdef HIP_COMPILE
#include "hip_legacy_executors.hpp"
#endif


/******************** fisheye ********************/

RppStatus rppi_fisheye_u8_pln1_batchPD_host(RppPtr_t srcPtr,
                                            RppiSize *srcSize,
                                            RppiSize maxSrcSize,
                                            RppPtr_t dstPtr,
                                            Rpp32u nbatchSize,
                                            rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_host_roi(roiPoints, rpp::deref(rppHandle));
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    fisheye_host_batch<Rpp8u>(static_cast<Rpp8u *>(srcPtr),
                              srcSize,
                              rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                              static_cast<Rpp8u *>(dstPtr),
                              rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                              rpp::deref(rppHandle).GetBatchSize(),
                              RPPI_CHN_PLANAR,
                              1,
                              rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus rppi_fisheye_u8_pkd3_batchPD_host(RppPtr_t srcPtr,
                                            RppiSize *srcSize,
                                            RppiSize maxSrcSize,
                                            RppPtr_t dstPtr,
                                            Rpp32u nbatchSize,
                                            rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_host_roi(roiPoints, rpp::deref(rppHandle));
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    fisheye_host_batch<Rpp8u>(static_cast<Rpp8u *>(srcPtr),
                              srcSize,
                              rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                              static_cast<Rpp8u *>(dstPtr),
                              rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                              rpp::deref(rppHandle).GetBatchSize(),
                              RPPI_CHN_PACKED,
                              3,
                              rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

/******************** snow ********************/

RppStatus rppi_snow_u8_pln1_batchPD_host(RppPtr_t srcPtr,
                                         RppiSize *srcSize,
                                         RppiSize maxSrcSize,
                                         RppPtr_t dstPtr,
                                         Rpp32f *snowValue,
                                         Rpp32u nbatchSize,
                                         rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_host_roi(roiPoints, rpp::deref(rppHandle));
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    snow_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                           srcSize,
                           rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                           static_cast<Rpp8u*>(dstPtr),
                           snowValue,
                           rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                           rpp::deref(rppHandle).GetBatchSize(),
                           RPPI_CHN_PLANAR,
                           1,
                           rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

RppStatus rppi_snow_u8_pkd3_batchPD_host(RppPtr_t srcPtr,
                                         RppiSize *srcSize,
                                         RppiSize maxSrcSize,
                                         RppPtr_t dstPtr,
                                         Rpp32f *snowValue,
                                         Rpp32u nbatchSize,
                                         rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_host_roi(roiPoints, rpp::deref(rppHandle));
    copy_host_maxSrcSize(maxSrcSize, rpp::deref(rppHandle));

    snow_host_batch<Rpp8u>(static_cast<Rpp8u*>(srcPtr),
                           srcSize,
                           rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.maxSrcSize,
                           static_cast<Rpp8u*>(dstPtr),
                           snowValue,
                           rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.roiPoints,
                           rpp::deref(rppHandle).GetBatchSize(),
                           RPPI_CHN_PACKED,
                           3,
                           rpp::deref(rppHandle));

    return RPP_SUCCESS;
}

/********************************************************************************************************************/
/*********************************************** RPP_GPU_SUPPORT = ON ***********************************************/
/********************************************************************************************************************/

#ifdef GPU_SUPPORT

/******************** fisheye ********************/
RppStatus rppi_fisheye_u8_pln1_batchPD_gpu(RppPtr_t srcPtr,
                                           RppiSize *srcSize,
                                           RppiSize maxSrcSize,
                                           RppPtr_t dstPtr,
                                           Rpp32u nbatchSize,
                                           rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);

    fisheye_hip_batch(static_cast<Rpp8u *>(srcPtr),
                        static_cast<Rpp8u *>(dstPtr),
                        rpp::deref(rppHandle),
                        RPPI_CHN_PLANAR,
                        1);

    return RPP_SUCCESS;
}


RppStatus rppi_fisheye_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr,
                                           RppiSize *srcSize,
                                           RppiSize maxSrcSize,
                                           RppPtr_t dstPtr,
                                           Rpp32u nbatchSize,
                                           rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

    fisheye_hip_batch(static_cast<Rpp8u *>(srcPtr),
                        static_cast<Rpp8u *>(dstPtr),
                        rpp::deref(rppHandle),
                        RPPI_CHN_PACKED,
                        3);


    return RPP_SUCCESS;
}

RppStatus rppi_snow_u8_pln1_batchPD_gpu(RppPtr_t srcPtr,
                                        RppiSize *srcSize,
                                        RppiSize maxSrcSize,
                                        RppPtr_t dstPtr,
                                        Rpp32f *snowValue,
                                        Rpp32u nbatchSize,
                                        rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 1, RPPI_CHN_PLANAR);
    copy_param_float(snowValue, rpp::deref(rppHandle), paramIndex++);

    snow_hip_batch(static_cast<Rpp8u*>(srcPtr),
                    static_cast<Rpp8u*>(dstPtr),
                    rpp::deref(rppHandle),
                    RPPI_CHN_PLANAR,
                    1);

    return RPP_SUCCESS;
}

RppStatus rppi_snow_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr,
                                        RppiSize *srcSize,
                                        RppiSize maxSrcSize,
                                        RppPtr_t dstPtr,
                                        Rpp32f *snowValue,
                                        Rpp32u nbatchSize,
                                        rppHandle_t rppHandle)
{
    RppiROI roiPoints;
    roiPoints.x = 0;
    roiPoints.y = 0;
    roiPoints.roiHeight = 0;
    roiPoints.roiWidth = 0;
    Rpp32u paramIndex = 0;
    copy_srcSize(srcSize, rpp::deref(rppHandle));
    copy_srcMaxSize(maxSrcSize, rpp::deref(rppHandle));
    copy_roi(roiPoints, rpp::deref(rppHandle));
    get_srcBatchIndex(rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);
    copy_param_float(snowValue, rpp::deref(rppHandle), paramIndex++);

    snow_hip_batch(static_cast<Rpp8u*>(srcPtr),
                    static_cast<Rpp8u*>(dstPtr),
                    rpp::deref(rppHandle),
                    RPPI_CHN_PACKED,
                    3);

    return RPP_SUCCESS;
}
#endif