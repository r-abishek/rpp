#include <cl/rpp_cl_common.hpp>
#include "cl_declarations.hpp"

/******************** non_linear_blend ********************/

RppStatus
non_linear_blend_cl_batch(cl_mem srcPtr1, cl_mem srcPtr2, cl_mem dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    int in_plnpkdind = getplnpkdind(tensor_info._in_format), out_plnpkdind = getplnpkdind(tensor_info._out_format);
    InitHandle *handle_obj = handle.GetInitHandle();
    Rpp32u max_height, max_width;
    max_size(handle_obj->mem.mgpu.csrcSize.height, handle_obj->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);
    std::vector<size_t> vld{16, 16, 1};
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};
    std::string kernel_file = "non_linear_blend.cl";
    std::string kernel_name = "non_linear_blend_batch";
    get_kernel_name(kernel_name, tensor_info);

    handle.AddKernel("", "", kernel_file, kernel_name, vld, vgd, "")(srcPtr1, srcPtr2, dstPtr,
                                                                     handle_obj->mem.mgpu.floatArr[0].floatmem,
                                                                     handle_obj->mem.mgpu.roiPoints.x,
                                                                     handle_obj->mem.mgpu.roiPoints.roiWidth,
                                                                     handle_obj->mem.mgpu.roiPoints.y,
                                                                     handle_obj->mem.mgpu.roiPoints.roiHeight,
                                                                     handle_obj->mem.mgpu.srcSize.height,
                                                                     handle_obj->mem.mgpu.srcSize.width,
                                                                     handle_obj->mem.mgpu.maxSrcSize.width,
                                                                     handle_obj->mem.mgpu.srcBatchIndex,
                                                                     tensor_info._in_channels,
                                                                     handle_obj->mem.mgpu.inc,
                                                                     handle_obj->mem.mgpu.dstInc,
                                                                     in_plnpkdind, out_plnpkdind);
    return RPP_SUCCESS;
}

/******************** water ********************/

RppStatus
water_cl_batch(cl_mem srcPtr, cl_mem dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    int in_plnpkdind = getplnpkdind(tensor_info._in_format), out_plnpkdind = getplnpkdind(tensor_info._out_format);
    InitHandle *handle_obj = handle.GetInitHandle();
    Rpp32u max_height, max_width;
    max_size(handle_obj->mem.mgpu.csrcSize.height, handle_obj->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);
    std::vector<size_t> vld{16, 16, 1};
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};
    std::string kernel_file = "water.cl";
    std::string kernel_name = "water_batch";
    get_kernel_name(kernel_name, tensor_info);

    handle.AddKernel("", "", kernel_file, kernel_name, vld, vgd, "")(srcPtr, dstPtr,
                                                                     handle_obj->mem.mgpu.floatArr[0].floatmem,
                                                                     handle_obj->mem.mgpu.floatArr[1].floatmem,
                                                                     handle_obj->mem.mgpu.floatArr[2].floatmem,
                                                                     handle_obj->mem.mgpu.floatArr[3].floatmem,
                                                                     handle_obj->mem.mgpu.floatArr[4].floatmem,
                                                                     handle_obj->mem.mgpu.floatArr[5].floatmem,
                                                                     handle_obj->mem.mgpu.roiPoints.x,
                                                                     handle_obj->mem.mgpu.roiPoints.roiWidth,
                                                                     handle_obj->mem.mgpu.roiPoints.y,
                                                                     handle_obj->mem.mgpu.roiPoints.roiHeight,
                                                                     handle_obj->mem.mgpu.srcSize.height,
                                                                     handle_obj->mem.mgpu.srcSize.width,
                                                                     handle_obj->mem.mgpu.maxSrcSize.height,
                                                                     handle_obj->mem.mgpu.maxSrcSize.width,
                                                                     handle_obj->mem.mgpu.srcBatchIndex,
                                                                     tensor_info._in_channels,
                                                                     handle_obj->mem.mgpu.inc,
                                                                     handle_obj->mem.mgpu.dstInc,
                                                                     in_plnpkdind, out_plnpkdind);
    return RPP_SUCCESS;
}

/******************** erase ********************/

RppStatus
erase_cl_batch(cl_mem srcPtr, cl_mem dstPtr, cl_mem anchor_box_info, cl_mem colors, cl_mem box_offset,
                             rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    int in_plnpkdind = getplnpkdind(tensor_info._in_format), out_plnpkdind = getplnpkdind(tensor_info._out_format);
    InitHandle *handle_obj = handle.GetInitHandle();
    Rpp32u max_height, max_width;
    max_size(handle_obj->mem.mgpu.csrcSize.height, handle_obj->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);
    std::vector<size_t> vld{16, 16, 1};
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};
    std::string kernel_file = "erase.cl";
    std::string kernel_name = "erase_batch";
    std::string kernel_pln1_name = "erase_pln1_batch";
    get_kernel_name(kernel_name, tensor_info);
    get_kernel_name(kernel_pln1_name, tensor_info);

    if (tensor_info._in_channels == 3)
        handle.AddKernel("", "", kernel_file, kernel_name, vld, vgd, "")(srcPtr, dstPtr, anchor_box_info, colors, box_offset,
                                                                         handle_obj->mem.mgpu.uintArr[0].uintmem,
                                                                         handle_obj->mem.mgpu.srcSize.height,
                                                                         handle_obj->mem.mgpu.srcSize.width,
                                                                         handle_obj->mem.mgpu.maxSrcSize.width,
                                                                         handle_obj->mem.mgpu.srcBatchIndex,
                                                                         handle_obj->mem.mgpu.inc,
                                                                         handle_obj->mem.mgpu.dstInc,
                                                                         in_plnpkdind, out_plnpkdind);
    else
        handle.AddKernel("", "", kernel_file, kernel_pln1_name, vld, vgd, "")(srcPtr, dstPtr, anchor_box_info, colors, box_offset,
                                                                        handle_obj->mem.mgpu.uintArr[0].uintmem,
                                                                        handle_obj->mem.mgpu.srcSize.height,
                                                                        handle_obj->mem.mgpu.srcSize.width,
                                                                        handle_obj->mem.mgpu.maxSrcSize.width,
                                                                        handle_obj->mem.mgpu.srcBatchIndex,
                                                                        handle_obj->mem.mgpu.inc,
                                                                        handle_obj->mem.mgpu.dstInc,
                                                                        in_plnpkdind, out_plnpkdind);

    return RPP_SUCCESS;
}

/******************** color_cast ********************/

RppStatus
color_cast_cl_batch(cl_mem srcPtr, cl_mem dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    int in_plnpkdind = getplnpkdind(tensor_info._in_format), out_plnpkdind = getplnpkdind(tensor_info._out_format);
    InitHandle *handle_obj = handle.GetInitHandle();
    Rpp32u max_height, max_width;
    max_size(handle_obj->mem.mgpu.csrcSize.height, handle_obj->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);
    std::vector<size_t> vld{16, 16, 1};
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};
    std::string kernel_file = "color_cast.cl";
    std::string kernel_name = "color_cast_batch";
    get_kernel_name(kernel_name, tensor_info);

    handle.AddKernel("", "", kernel_file, kernel_name, vld, vgd, "")(srcPtr, dstPtr,
                                                                     handle_obj->mem.mgpu.ucharArr[0].ucharmem,
                                                                     handle_obj->mem.mgpu.ucharArr[1].ucharmem,
                                                                     handle_obj->mem.mgpu.ucharArr[2].ucharmem,
                                                                     handle_obj->mem.mgpu.floatArr[3].floatmem,
                                                                     handle_obj->mem.mgpu.roiPoints.x,
                                                                     handle_obj->mem.mgpu.roiPoints.roiWidth,
                                                                     handle_obj->mem.mgpu.roiPoints.y,
                                                                     handle_obj->mem.mgpu.roiPoints.roiHeight,
                                                                     handle_obj->mem.mgpu.srcSize.height,
                                                                     handle_obj->mem.mgpu.srcSize.width,
                                                                     handle_obj->mem.mgpu.maxSrcSize.width,
                                                                     handle_obj->mem.mgpu.srcBatchIndex,
                                                                     tensor_info._in_channels,
                                                                     handle_obj->mem.mgpu.inc,
                                                                     handle_obj->mem.mgpu.dstInc,
                                                                     in_plnpkdind, out_plnpkdind);
    return RPP_SUCCESS;
}

/******************** lut ********************/

RppStatus
lut_cl_batch(cl_mem srcPtr, cl_mem dstPtr, cl_mem lut, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    int in_plnpkdind = getplnpkdind(tensor_info._in_format), out_plnpkdind = getplnpkdind(tensor_info._out_format);
    InitHandle *handle_obj = handle.GetInitHandle();
    Rpp32u max_height, max_width;
    max_size(handle_obj->mem.mgpu.csrcSize.height, handle_obj->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);
    std::vector<size_t> vld{16, 16, 1};
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};
    std::string kernel_file = "look_up_table.cl";
    std::string kernel_name = "look_up_table_batch_tensor";
    get_kernel_name(kernel_name, tensor_info);

    handle.AddKernel("", "", kernel_file, kernel_name, vld, vgd, "")(srcPtr, dstPtr, lut,
                                                                     handle_obj->mem.mgpu.srcSize.height,
                                                                     handle_obj->mem.mgpu.srcSize.width,
                                                                     handle_obj->mem.mgpu.maxSrcSize.width,
                                                                     handle_obj->mem.mgpu.srcBatchIndex,
                                                                     tensor_info._in_channels,
                                                                     handle_obj->mem.mgpu.inc,
                                                                     handle_obj->mem.mgpu.dstInc,
                                                                     in_plnpkdind, out_plnpkdind);

    return RPP_SUCCESS;
}

/******************** crop_and_patch ********************/

RppStatus
crop_and_patch_cl_batch(cl_mem srcPtr1, cl_mem srcPtr2, cl_mem dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    int in_plnpkdind = getplnpkdind(tensor_info._in_format), out_plnpkdind = getplnpkdind(tensor_info._out_format);
    InitHandle *handle_obj = handle.GetInitHandle();
    Rpp32u max_height, max_width;
    max_size(handle_obj->mem.mgpu.csrcSize.height, handle_obj->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);
    std::vector<size_t> vld{16, 16, 1};
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};
    std::string kernel_file = "crop_and_patch.cl";
    std::string kernel_name = "crop_and_patch_batch";
    get_kernel_name(kernel_name, tensor_info);

    handle.AddKernel("", "", kernel_file, kernel_name, vld, vgd, "")(srcPtr1, srcPtr2 , dstPtr,
                                                                     handle_obj->mem.mgpu.srcSize.height,
                                                                     handle_obj->mem.mgpu.srcSize.width,
                                                                     handle_obj->mem.mgpu.dstSize.height,
                                                                     handle_obj->mem.mgpu.dstSize.width,
                                                                     handle_obj->mem.mgpu.uintArr[4].uintmem,
                                                                     handle_obj->mem.mgpu.uintArr[5].uintmem,
                                                                     handle_obj->mem.mgpu.uintArr[6].uintmem,
                                                                     handle_obj->mem.mgpu.uintArr[7].uintmem,
                                                                     handle_obj->mem.mgpu.uintArr[0].uintmem,
                                                                     handle_obj->mem.mgpu.uintArr[1].uintmem,
                                                                     handle_obj->mem.mgpu.uintArr[2].uintmem,
                                                                     handle_obj->mem.mgpu.uintArr[3].uintmem,
                                                                     handle_obj->mem.mgpu.maxSrcSize.width,
                                                                     handle_obj->mem.mgpu.maxDstSize.width,
                                                                     handle_obj->mem.mgpu.srcBatchIndex,
                                                                     handle_obj->mem.mgpu.dstBatchIndex,
                                                                     tensor_info._in_channels,
                                                                     handle_obj->mem.mgpu.inc,
                                                                     handle_obj->mem.mgpu.dstInc,
                                                                     in_plnpkdind, out_plnpkdind);

    return RPP_SUCCESS;
}

/******************** glitch ********************/

RppStatus
glitch_cl_batch(cl_mem srcPtr, cl_mem dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    int in_plnpkdind = getplnpkdind(tensor_info._in_format), out_plnpkdind = getplnpkdind(tensor_info._out_format);
    InitHandle *handle_obj = handle.GetInitHandle();
    Rpp32u max_height, max_width;
    max_size(handle_obj->mem.mgpu.csrcSize.height, handle_obj->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);
    std::vector<size_t> vld{16, 16, 1};
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};
    std::string kernel_file = "glitch.cl";
    std::string kernel_name = "glitch_batch";
    get_kernel_name(kernel_name, tensor_info);

    handle.AddKernel("", "", kernel_file, kernel_name, vld, vgd, "")(srcPtr, dstPtr,
                                                                     handle_obj->mem.mgpu.uintArr[0].uintmem,
                                                                     handle_obj->mem.mgpu.uintArr[1].uintmem,
                                                                     handle_obj->mem.mgpu.uintArr[2].uintmem,
                                                                     handle_obj->mem.mgpu.uintArr[3].uintmem,
                                                                     handle_obj->mem.mgpu.uintArr[4].uintmem,
                                                                     handle_obj->mem.mgpu.uintArr[5].uintmem,
                                                                     handle_obj->mem.mgpu.roiPoints.x,
                                                                     handle_obj->mem.mgpu.roiPoints.roiWidth,
                                                                     handle_obj->mem.mgpu.roiPoints.y,
                                                                     handle_obj->mem.mgpu.roiPoints.roiHeight,
                                                                     handle_obj->mem.mgpu.srcSize.height,
                                                                     handle_obj->mem.mgpu.srcSize.width,
                                                                     handle_obj->mem.mgpu.maxSrcSize.width,
                                                                     handle_obj->mem.mgpu.srcBatchIndex,
                                                                     tensor_info._in_channels,
                                                                     handle_obj->mem.mgpu.inc,
                                                                     handle_obj->mem.mgpu.dstInc,
                                                                     in_plnpkdind, out_plnpkdind);

    return RPP_SUCCESS;
}

// /******************** ricap ********************/ // TODO : To be verifed later

// RppStatus
// ricap_cl_batch(cl_mem srcPtr, cl_mem dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
// {
//     int in_plnpkdind = getplnpkdind(tensor_info._in_format), out_plnpkdind = getplnpkdind(tensor_info._out_format);
//     InitHandle *handle_obj = handle.GetInitHandle();
//     Rpp32u max_height, max_width;
//     max_size(handle_obj->mem.mgpu.csrcSize.height, handle_obj->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);
//     std::vector<size_t> vld{16, 16, 1};
//     std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};
//     std::string kernel_file = "ricap.cl";
//     std::string kernel_name = "ricap_batch";
//     get_kernel_name(kernel_name, tensor_info);

//     handle.AddKernel("", "", kernel_file, kernel_name, vld, vgd, "")(srcPtr, dstPtr,
//                                                                      handle_obj->mem.mgpu.srcSize.height,
//                                                                      handle_obj->mem.mgpu.srcSize.width,
//                                                                      handle_obj->mem.mgpu.dstSize.height,
//                                                                      handle_obj->mem.mgpu.dstSize.width,
//                                                                      handle_obj->mem.mgpu.uintArr[4].uintmem,
//                                                                      handle_obj->mem.mgpu.uintArr[5].uintmem,
//                                                                      handle_obj->mem.mgpu.uintArr[6].uintmem,
//                                                                      handle_obj->mem.mgpu.uintArr[7].uintmem,
//                                                                      handle_obj->mem.mgpu.uintArr[0].uintmem,
//                                                                      handle_obj->mem.mgpu.uintArr[1].uintmem,
//                                                                      handle_obj->mem.mgpu.uintArr[2].uintmem,
//                                                                      handle_obj->mem.mgpu.uintArr[3].uintmem,
//                                                                      handle_obj->mem.mgpu.maxSrcSize.width,
//                                                                      handle_obj->mem.mgpu.maxDstSize.width,
//                                                                      handle_obj->mem.mgpu.srcBatchIndex,
//                                                                      handle_obj->mem.mgpu.dstBatchIndex,
//                                                                      tensor_info._in_channels,
//                                                                      handle_obj->mem.mgpu.inc,
//                                                                      handle_obj->mem.mgpu.dstInc,
//                                                                      in_plnpkdind, out_plnpkdind);

//     return RPP_SUCCESS;
// }