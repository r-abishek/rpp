#include <rpp.h>
#include <stdlib.h>

#include <iostream>
#include <string>
#include <vector>
#include <numeric>
#include <unordered_map>
#include <chrono>
#include <thread>
#include <fstream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>

#include "optical_flow_vectors_2.hpp"
// #include "inputs.hpp"
#include "inputsStrided.hpp"

using namespace cv;
using namespace std;
using namespace std::chrono;

#define FARNEBACK_FRAME_WIDTH 960                       // Farneback algorithm frame width
#define FARNEBACK_FRAME_HEIGHT 540                      // Farneback algorithm frame height
#define FARNEBACK_OUTPUT_FRAME_SIZE 518400u             // 960 * 540
#define FARNEBACK_OUTPUT_MOTION_VECTORS_SIZE 1036800u   // 960 * 540 * 2
#define FARNEBACK_OUTPUT_RGB_SIZE 1555200u              // 960 * 540 * 3
#define HUE_CONVERSION_FACTOR 0.0019607843f             // ((1 / 360.0) * (180 / 255.0))

const RpptInterpolationType interpolationType = RpptInterpolationType::NEAREST_NEIGHBOR;
const RpptSubpixelLayout subpixelLayout = RpptSubpixelLayout::BGRtype;
const RpptRoiType roiType = RpptRoiType::XYWH;
const RpptAngleType angleType = RpptAngleType::DEGREES;

void rpp_tensor_initialize_descriptor(RpptDescPtr descPtr,
                                      RpptLayout layout,
                                      RpptDataType dataType,
                                      Rpp32u offsetInBytes,
                                      RppSize_t numDims,
                                      Rpp32u n,
                                      Rpp32u h,
                                      Rpp32u w,
                                      Rpp32u c,
                                      Rpp32u nStride,
                                      Rpp32u hStride,
                                      Rpp32u wStride,
                                      Rpp32u cStride)
{
    descPtr->layout = layout;
    descPtr->dataType = dataType;
    descPtr->offsetInBytes = offsetInBytes;
    descPtr->numDims = numDims;
    descPtr->n = n;
    descPtr->h = h;
    descPtr->w = w;
    descPtr->c = c;
    descPtr->strides.nStride = nStride;
    descPtr->strides.hStride = hStride;
    descPtr->strides.wStride = wStride;
    descPtr->strides.cStride = cStride;
}

void rpp_tensor_initialize_descriptor_generic(RpptGenericDescPtr descPtr,
                                              RpptLayout layout,
                                              RpptDataType dataType,
                                              Rpp32u offsetInBytes,
                                              RppSize_t numDims,
                                              Rpp32u dim0,
                                              Rpp32u dim1,
                                              Rpp32u dim2,
                                              Rpp32u dim3,
                                              Rpp32u stride0,
                                              Rpp32u stride1,
                                              Rpp32u stride2,
                                              Rpp32u stride3)
{
    descPtr->layout = layout;
    descPtr->dataType = dataType;
    descPtr->offsetInBytes = offsetInBytes;
    descPtr->numDims = numDims;
    descPtr->dims[0] = dim0;
    descPtr->dims[1] = dim1;
    descPtr->dims[2] = dim2;
    descPtr->dims[3] = dim3;
    descPtr->strides[0] = stride0;
    descPtr->strides[1] = stride1;
    descPtr->strides[2] = stride2;
    descPtr->strides[3] = stride3;
}

void rpp_tensor_initialize_roi_xywh(RpptROIPtr roiPtr, Rpp32u x, Rpp32u y, Rpp32u roiWidth, Rpp32u roiHeight)
{
    roiPtr->xywhROI.xy.x = x;
    roiPtr->xywhROI.xy.y = y;
    roiPtr->xywhROI.roiWidth = roiWidth;
    roiPtr->xywhROI.roiHeight = roiHeight;
}

void rpp_tensor_create_strided(Rpp8u *frame_u8, Rpp8u *srcRGB, RpptDescPtr srcDescPtrRGB)
{
    Rpp8u *frameTemp_u8 = frame_u8;
    Rpp8u *srcRGBTemp = srcRGB + srcDescPtrRGB->offsetInBytes;
    Rpp32u elementsInRow = srcDescPtrRGB->c * srcDescPtrRGB->w;
    for (int i = 0; i < srcDescPtrRGB->h; i++)
    {
        memcpy(srcRGBTemp, frameTemp_u8, elementsInRow);
        srcRGBTemp += srcDescPtrRGB->strides.hStride;
        frameTemp_u8 += elementsInRow;
    }
}

template <typename T>
void rpp_tensor_print(string tensorName, T *tensor, RpptDescPtr desc)
{
    std::cout << "\n" << tensorName << ":\n";
    T *tensorTemp;
    tensorTemp = tensor + desc->offsetInBytes;
    for (int n = 0; n < desc->n; n++)
    {
        std::cout << "[ ";
        for (int h = 0; h < desc->h; h++)
        {
            for (int w = 0; w < desc->w; w++)
            {
                for (int c = 0; c < desc->c; c++)
                {
                    std::cout << (Rpp32f)*tensorTemp << ", ";
                    tensorTemp++;
                }
            }
            std::cout << ";\n";
        }
        std::cout << " ]\n\n";
    }
}

template <typename T>
void rpp_tensor_write_to_file(string tensorName, T *tensor, RpptDescPtr desc)
{
    ofstream outputFile (tensorName + ".csv");

    outputFile << "\n" << tensorName << ":\n";
    T *tensorTemp;
    tensorTemp = tensor + desc->offsetInBytes;
    for (int n = 0; n < desc->n; n++)
    {
        outputFile << "[ ";
        for (int h = 0; h < desc->h; h++)
        {
            for (int w = 0; w < desc->w; w++)
            {
                for (int c = 0; c < desc->c; c++)
                {
                    outputFile << (Rpp32f)*tensorTemp << ", ";
                    tensorTemp++;
                }
            }
            outputFile << ";\n";
        }
        outputFile << " ]\n\n";
    }
}

void rpp_tensor_write_to_images(string tensorName, Rpp8u *tensor, RpptDescPtr desc, RpptImagePatch *imgSizes)
{
    Rpp8u outputImage_u8[desc->strides.nStride];
    Rpp8u *tensorTemp, *outputImageTemp_u8;
    tensorTemp = tensor + desc->offsetInBytes;
    outputImageTemp_u8 = outputImage_u8;

    for (int n = 0; n < desc->n; n++)
    {
        Rpp32u elementsInRow = desc->c * imgSizes[n].width;
        for (int i = 0; i < imgSizes[n].height; i++)
        {
            memcpy(outputImageTemp_u8, tensorTemp, elementsInRow);
            tensorTemp += desc->strides.hStride;
            outputImageTemp_u8 += elementsInRow;
        }

        cv::Mat outputImage_mat;
        outputImage_mat = (desc->c == 1) ? Mat(imgSizes[n].height, imgSizes[n].width, CV_8UC1, outputImage_u8) : Mat(imgSizes[n].height, imgSizes[n].width, CV_8UC3, outputImage_u8);
        imwrite(tensorName + "_" + std::to_string(n) + ".jpg", outputImage_mat);
    }
}

void rpp_tensor_write_to_images_and_append_frame_to_video(string tensorName, Rpp8u *tensor, RpptDescPtr desc, RpptImagePatch *imgSizes, cv::VideoWriter &videoOutput, int iterCount)
{
    Rpp8u outputImage_u8[desc->strides.nStride];
    Rpp8u *tensorTemp, *outputImageTemp_u8;
    tensorTemp = tensor + desc->offsetInBytes;
    outputImageTemp_u8 = outputImage_u8;

    for (int n = 0; n < desc->n; n++)
    {
        Rpp32u elementsInRow = desc->c * imgSizes[n].width;
        for (int i = 0; i < imgSizes[n].height; i++)
        {
            memcpy(outputImageTemp_u8, tensorTemp, elementsInRow);
            tensorTemp += desc->strides.hStride;
            outputImageTemp_u8 += elementsInRow;
        }

        cv::Mat outputImage_mat;
        outputImage_mat = (desc->c == 1) ? Mat(imgSizes[n].height, imgSizes[n].width, CV_8UC1, outputImage_u8) : Mat(imgSizes[n].height, imgSizes[n].width, CV_8UC3, outputImage_u8);
        // imwrite(tensorName + "_" + std::to_string(n) + "_" + std::to_string(iterCount) + ".jpg", outputImage_mat);
        videoOutput.write(outputImage_mat);
    }
}

void rpp_optical_flow_hip(string inputVideoFileName)
{
    std::cerr << "\n->>>>>>>DEBUG>>>>>>>>> Entering rpp_optical_flow_hip()";

    // initialize map to track time for every stage at each iteration
    unordered_map<string, vector<double>> timers;

    // initialize video capture with opencv video
    VideoCapture capture(inputVideoFileName);
    if (!capture.isOpened())
    {
        // error in opening the video file
        cout << "\nUnable to open file!";
        return;
    }

    // get video properties
    double fps = capture.get(CAP_PROP_FPS);                     // input video fps
    int numOfFrames = int(capture.get(CAP_PROP_FRAME_COUNT));   // input video number of frames
    int frameWidth = int(capture.get(CAP_PROP_FRAME_WIDTH));    // input video frame width
    int frameHeight = int(capture.get(CAP_PROP_FRAME_HEIGHT));  // input video frame height
    int bitRate = int(capture.get(CAP_PROP_BITRATE));           // input video bitrate

    // declare rpp tensor descriptors
    RpptDesc srcDescRGB, srcDesc, dstDescHSV, dstDescRGB, mVecCompPlnDesc;
    RpptDescPtr srcDescPtrRGB, srcDescPtr, dstDescPtrHSV, dstDescPtrRGB, mVecCompPlnDescPtr;
    RpptGenericDesc mVecPolrPlnGenericDesc, mVecCartPlnGenericDesc, mVecCompPlnGenericDesc;
    RpptGenericDescPtr mVecPolrPlnGenericDescPtr, mVecCartPlnGenericDescPtr, mVecCompPlnGenericDescPtr;
    srcDescPtrRGB = &srcDescRGB;
    srcDescPtr = &srcDesc;
    dstDescPtrHSV = &dstDescHSV;
    dstDescPtrRGB = &dstDescRGB;
    mVecCompPlnDescPtr = &mVecCompPlnDesc;
    mVecCartPlnGenericDescPtr = &mVecCartPlnGenericDesc;
    mVecPolrPlnGenericDescPtr = &mVecPolrPlnGenericDesc;
    mVecCompPlnGenericDescPtr = &mVecCompPlnGenericDesc;

    // initialize 8 different rpp tensor descriptors used in the whole optical flow pipeline
    // initialize rpp tensor descriptor for srcRGB frames
    rpp_tensor_initialize_descriptor(srcDescPtrRGB, RpptLayout::NHWC, RpptDataType::U8, 0, 4, 1, frameHeight, frameWidth, 3, frameHeight * frameWidth * 3, frameWidth * 3, 3, 1);
    // initialize rpp tensor descriptor for dstRGB frames (same descriptor as srcRGB except with a resized farneback width/height and stride changes)
    rpp_tensor_initialize_descriptor(dstDescPtrRGB, RpptLayout::NHWC, RpptDataType::U8, 0, 4, 1, FARNEBACK_FRAME_HEIGHT, FARNEBACK_FRAME_WIDTH, 3, FARNEBACK_OUTPUT_RGB_SIZE, FARNEBACK_FRAME_WIDTH * 3, 3, 1);
    // initialize rpp tensor descriptors for src greyscale frames (layout changed to NCHW, single channel)
    rpp_tensor_initialize_descriptor(srcDescPtr, RpptLayout::NCHW, RpptDataType::U8, 0, 4, 1, FARNEBACK_FRAME_HEIGHT, FARNEBACK_FRAME_WIDTH, 1, FARNEBACK_OUTPUT_FRAME_SIZE, FARNEBACK_FRAME_WIDTH, 1, FARNEBACK_OUTPUT_FRAME_SIZE);
    // initialize rpp tensor descriptors for dstHSV NCHW PLN3 F32 frames
    rpp_tensor_initialize_descriptor(dstDescPtrHSV, RpptLayout::NCHW, RpptDataType::F32, 0, 4, 1, FARNEBACK_FRAME_HEIGHT, FARNEBACK_FRAME_WIDTH, 3, FARNEBACK_OUTPUT_RGB_SIZE, FARNEBACK_FRAME_WIDTH, 1, FARNEBACK_OUTPUT_FRAME_SIZE);
    // initialize rpp tensor descriptor to extract one component of motion vector
    rpp_tensor_initialize_descriptor(mVecCompPlnDescPtr, RpptLayout::NCHW, RpptDataType::F32, 0, 4, 1, FARNEBACK_FRAME_HEIGHT, FARNEBACK_FRAME_WIDTH, 1, FARNEBACK_OUTPUT_FRAME_SIZE, FARNEBACK_FRAME_WIDTH, 1, FARNEBACK_OUTPUT_FRAME_SIZE);
    // initialize rpp generic tensor descriptor for planar cartesian motion vectors (a 2 channel NCHW tensor)
    rpp_tensor_initialize_descriptor_generic(mVecCartPlnGenericDescPtr, RpptLayout::NCHW, RpptDataType::F32, 0, 4, 1, 2, FARNEBACK_FRAME_HEIGHT, FARNEBACK_FRAME_WIDTH, FARNEBACK_OUTPUT_FRAME_SIZE * 2, FARNEBACK_OUTPUT_FRAME_SIZE, FARNEBACK_FRAME_WIDTH, 1);
    // initialize rpp generic tensor descriptor for planar polar motion vectors (a 2 channel NCHW tensor)
    rpp_tensor_initialize_descriptor_generic(mVecPolrPlnGenericDescPtr, RpptLayout::NCHW, RpptDataType::F32, 0, 4, 1, 2, FARNEBACK_FRAME_HEIGHT, FARNEBACK_FRAME_WIDTH, FARNEBACK_OUTPUT_FRAME_SIZE * 4, FARNEBACK_OUTPUT_FRAME_SIZE, FARNEBACK_FRAME_WIDTH, 1);
    // initialize rpp generic tensor descriptor to extract one component of motion vector
    rpp_tensor_initialize_descriptor_generic(mVecCompPlnGenericDescPtr, RpptLayout::NCHW, RpptDataType::F32, 0, 4, 1, 1, FARNEBACK_FRAME_HEIGHT, FARNEBACK_FRAME_WIDTH, FARNEBACK_OUTPUT_FRAME_SIZE, FARNEBACK_OUTPUT_FRAME_SIZE, FARNEBACK_FRAME_WIDTH, 1);

    std::cerr << "\n->>>>>>>DEBUG>>>>>>>>> checkpoint 1";

    // TEMP initialization to dump saved CSV of motion vectors
    RpptGenericDesc mVecCartPkdGenericDesc;
    RpptGenericDescPtr mVecCartPkdGenericDescPtr;
    mVecCartPkdGenericDescPtr = &mVecCartPkdGenericDesc;
    // initialize rpp generic tensor descriptor for packed cartesian motion vectors (a 2 channel NHWC tensor)
    rpp_tensor_initialize_descriptor_generic(mVecCartPkdGenericDescPtr, RpptLayout::NHWC, RpptDataType::F32, 0, 4, 1, FARNEBACK_FRAME_HEIGHT, FARNEBACK_FRAME_WIDTH, 2, FARNEBACK_OUTPUT_FRAME_SIZE * 2, FARNEBACK_FRAME_WIDTH * 2, 2, 1);

    // set rpp tensor buffer sizes in bytes for srcRGB, dstRGB, src frames in src1 and src2
    unsigned long long sizeInBytesSrcRGB, sizeInBytesDstRGB, sizeInBytesSrc;
    sizeInBytesSrcRGB = (srcDescPtrRGB->n * srcDescPtrRGB->strides.nStride) + srcDescPtrRGB->offsetInBytes;
    sizeInBytesDstRGB = (dstDescPtrRGB->n * dstDescPtrRGB->strides.nStride) + dstDescPtrRGB->offsetInBytes;
    sizeInBytesSrc = (2 * srcDescPtr->n * srcDescPtr->strides.nStride) + srcDescPtr->offsetInBytes;

    // allocate rpp 8u host and hip buffers for srcRGB, dstRGB, dstInputRGB, src1 and src2
    Rpp8u *srcRGB = (Rpp8u *)calloc(sizeInBytesSrcRGB, 1);
    Rpp8u *dstRGB = (Rpp8u *)calloc(sizeInBytesDstRGB, 1);
    Rpp8u *dstInputRGB = (Rpp8u *)calloc(sizeInBytesDstRGB, 1);
    Rpp8u *srcBufs = (Rpp8u *)calloc(sizeInBytesSrc + 64, 1);
    Rpp8u *src1 = srcBufs + 64;
    Rpp8u *src2 = src1 + (srcDescPtr->n * srcDescPtr->strides.nStride) + srcDescPtr->offsetInBytes;
    Rpp8u *d_srcRGB, *d_dstRGB, *d_srcBufs, *d_src1, *d_src2;
    hipMalloc(&d_srcRGB, sizeInBytesSrcRGB);
    hipMalloc(&d_dstRGB, sizeInBytesDstRGB);
    hipMalloc(&d_srcBufs, sizeInBytesSrc + 64);
    d_src1 = d_srcBufs + 64;
    // Rpp32u num = (srcDescPtr->n * srcDescPtr->strides.nStride) + srcDescPtr->offsetInBytes;
    d_src2 = d_src1 + srcDescPtr->strides.nStride;
    // std::cerr << "\n->>>>>>>DEBUG>>>>>>>>> num = " << num;
    std::cerr << "\n->>>>>>>DEBUG>>>>>>>>> checkpoint 2";

    // allocate and initialize rpp roi and imagePatch buffers for resize ops on pinned memory
    RpptROI *roiTensorPtrSrcRGB, *roiTensorPtrDstRGB;
    RpptImagePatch *fbackImgPatchPtr;
    hipHostMalloc(&roiTensorPtrSrcRGB, srcDescPtrRGB->n * sizeof(RpptROI));
    std::cerr << "\n->>>>>>>DEBUG>>>>>>>>> checkpoint 2a";
    hipHostMalloc(&roiTensorPtrDstRGB, dstDescPtrRGB->n * sizeof(RpptROI));
    std::cerr << "\n->>>>>>>DEBUG>>>>>>>>> checkpoint 2b";
    hipHostMalloc(&fbackImgPatchPtr, srcDescPtrRGB->n * sizeof(RpptImagePatch));
    std::cerr << "\n->>>>>>>DEBUG>>>>>>>>> checkpoint 2c";
    *roiTensorPtrSrcRGB = {0, 0, srcDescPtrRGB->w, srcDescPtrRGB->h};
    // rpp_tensor_initialize_roi_xywh(roiTensorPtrSrcRGB, 0, 0, srcDescPtrRGB->w, srcDescPtrRGB->h);
    std::cerr << "\n->>>>>>>DEBUG>>>>>>>>> checkpoint 2d";
    *roiTensorPtrDstRGB = {0, 0, FARNEBACK_FRAME_WIDTH, FARNEBACK_FRAME_HEIGHT};
    // rpp_tensor_initialize_roi_xywh(roiTensorPtrDstRGB, 0, 0, FARNEBACK_FRAME_WIDTH, FARNEBACK_FRAME_HEIGHT);
    std::cerr << "\n->>>>>>>DEBUG>>>>>>>>> checkpoint 2e";
    fbackImgPatchPtr[0].width = dstDescPtrRGB->w;
    std::cerr << "\n->>>>>>>DEBUG>>>>>>>>> checkpoint 2f";
    fbackImgPatchPtr[0].height = dstDescPtrRGB->h;

    std::cerr << "\n->>>>>>>DEBUG>>>>>>>>> checkpoint 2g";

    // allocate hip motion vector buffers
    Rpp32f *d_motionVectorsCartesianF32, *d_motionVectorsCartesianF32Comp1, *d_motionVectorsCartesianF32Comp2;
    Rpp32f *d_motionVectorsPolarF32, *d_motionVectorsPolarF32Comp1, *d_motionVectorsPolarF32Comp2, *d_motionVectorsPolarF32Comp3;
    hipMalloc(&d_motionVectorsCartesianF32, FARNEBACK_OUTPUT_MOTION_VECTORS_SIZE * sizeof(Rpp32f));
    hipMalloc(&d_motionVectorsPolarF32, FARNEBACK_OUTPUT_FRAME_SIZE * 4 * sizeof(Rpp32f));
    d_motionVectorsCartesianF32Comp1 = d_motionVectorsCartesianF32;
    d_motionVectorsCartesianF32Comp2 = d_motionVectorsCartesianF32Comp1 + FARNEBACK_OUTPUT_FRAME_SIZE;
    d_motionVectorsPolarF32Comp1 = d_motionVectorsPolarF32 + FARNEBACK_OUTPUT_FRAME_SIZE;
    d_motionVectorsPolarF32Comp2 = d_motionVectorsPolarF32Comp1 + FARNEBACK_OUTPUT_FRAME_SIZE;
    d_motionVectorsPolarF32Comp3 = d_motionVectorsPolarF32Comp2 + FARNEBACK_OUTPUT_FRAME_SIZE;

    std::cerr << "\n->>>>>>>DEBUG>>>>>>>>> checkpoint 3";
    // preinitialize saturation channel portion of the buffer for HSV and reuse on every iteration in post-processing
    Rpp32f saturationChannel[FARNEBACK_OUTPUT_FRAME_SIZE];
    std::fill(&saturationChannel[0], &saturationChannel[FARNEBACK_OUTPUT_FRAME_SIZE - 1], 1.0f);
    hipMemcpy(d_motionVectorsPolarF32Comp2, saturationChannel, FARNEBACK_OUTPUT_FRAME_SIZE * sizeof(Rpp32f), hipMemcpyHostToDevice);

    // allocate post-processing buffer for imageMinMax
    Rpp32f *imageMinMaxArr;
    Rpp32u imageMinMaxArrLength = 2 * srcDescPtrRGB->n;
    hipHostMalloc(&imageMinMaxArr, imageMinMaxArrLength * sizeof(Rpp32f));

    // declare cpu outputs for optical flow
    // cv::Mat hsv[3], angle, bgr;      // ------- revisit

    // declare gpu outputs for optical flow
    // cv::cuda::GpuMat gpuMagnitude, gpuNormalizedMagnitude, gpuAngle;      // ------- revisit
    // cv::cuda::GpuMat gpuHSV[3], gpuMergedHSV, gpuHSV_8u, gpuBGR;      // ------- revisit

    // set saturation to 1
    // hsv[1] = cv::Mat::ones(frame.size(), CV_32F);      // ------- revisit
    // gpuHSV[1].upload(hsv[1]);      // ------- revisit
    std::cerr << "\n->>>>>>>DEBUG>>>>>>>>> checkpoint 4";

    // create rpp handle for hip with stream and batch size
    rppHandle_t handle1, handle2;
    hipStream_t stream1, stream2;
    hipStreamCreate(&stream1);
    hipStreamCreate(&stream2);
    rppCreateWithStreamAndBatchSize(&handle1, stream1, srcDescPtrRGB->n);
    rppCreateWithStreamAndBatchSize(&handle2, stream2, srcDescPtrRGB->n);

    // read the first frame
    cv::Mat frame, previousFrame;
    capture >> frame;
    // frame = Mat(720, 1200, CV_8UC3, cv::Scalar(255, 255, 255));
    // rectangle(
    //     frame,
    //     Point(100, 100),
    //     Point(300, 300),
    //     Scalar(0, 0, 0),
    //     FILLED,
    //     LINE_8);
    // frame = cv::imread("../../TEST_VIDEO/black_image_with_box.png", 1);
    // cv::imwrite("frame0.png", frame);

    // copy frame into rpp hip buffer
    // rpp_tensor_create_strided(frame.data, srcRGB, srcDescPtrRGB);    // not required since farneback frame size is already multiple of 8
    hipMemcpy(d_srcRGB, frame.data, sizeInBytesSrcRGB, hipMemcpyHostToDevice);

    // -------------------- stage output dump check --------------------
    // rpp_tensor_write_to_file("srcRGB", srcRGB, srcDescPtrRGB);
    // RpptImagePatch srcRGBImgSizes;
    // srcRGBImgSizes.height = srcDescPtrRGB->h;
    // srcRGBImgSizes.width = srcDescPtrRGB->w;
    // rpp_tensor_write_to_images("srcRGB", srcRGB, srcDescPtrRGB, &srcRGBImgSizes);

    // resize frame
    rppt_resize_gpu(d_srcRGB, srcDescPtrRGB, d_dstRGB, dstDescPtrRGB, fbackImgPatchPtr, interpolationType, roiTensorPtrSrcRGB, roiType, handle1);

    // -------------------- stage output dump check --------------------
    hipDeviceSynchronize();
    // hipMemcpy(dstRGB, d_dstRGB, sizeInBytesDstRGB, hipMemcpyDeviceToHost);
    // hipDeviceSynchronize();
    // rpp_tensor_write_to_file("srcResizedRGB1", dstRGB, dstDescPtrRGB);
    // rpp_tensor_write_to_images("srcResizedRGB1", dstRGB, dstDescPtrRGB, fbackImgPatchPtr);

    // convert to gray
    rppt_color_to_greyscale_gpu(d_dstRGB, dstDescPtrRGB, d_src1, srcDescPtr, subpixelLayout, handle1);

    // TEMP copy cuda src buffer 1
    // hipMemcpy(d_src1, inputs, FARNEBACK_OUTPUT_FRAME_SIZE, hipMemcpyHostToDevice);

    // -------------------- stage output dump check --------------------
    hipDeviceSynchronize();
    // hipMemcpy(src1, d_src1, sizeInBytesSrc / 2, hipMemcpyDeviceToHost);
    // rpp_tensor_write_to_file("src1", src1, srcDescPtr);
    // rpp_tensor_write_to_images("src1", src1, srcDescPtr, fbackImgPatchPtr);

    Size frameSize(960, 540);
    cv::VideoWriter videoOutput("videoOutput.mp4", VideoWriter::fourcc('M', 'J', 'P', 'G'), 15, frameSize);

    std::cerr << "\n->>>>>>>DEBUG>>>>>>>>> Enterning frame loop";

    int iterCount = 0;
    while (true)
    {
        std::cerr << "\n->>>>>>>>>>>>>>>>>>>>>>>> Iter# " << iterCount++;
        // start full pipeline timer
        auto startFullTime = high_resolution_clock::now();

        // ****************************************************************** reading ******************************************************************

        // start reading timer
        auto startReadTime = high_resolution_clock::now();

        // capture frame-by-frame
        capture >> frame;
        // frame = Mat(720, 1200, CV_8UC3, cv::Scalar(255, 255, 255));
        // rectangle(
        //     frame,
        //     Point(110, 100),
        //     Point(310, 300),
        //     Scalar(0, 0, 0),
        //     FILLED,
        //     LINE_8);
        // frame = cv::imread("../../TEST_VIDEO/black_image_with_box_1.png", 1);
        // cv::imwrite("frame1.png", frame);

        if (frame.empty())
            break;

        // copy frame into rpp hip buffer
        // rpp_tensor_create_strided(frame.data, srcRGB, srcDescPtrRGB);    // not required since farneback frame size is already multiple of 8
        hipMemcpy(d_srcRGB, frame.data, sizeInBytesSrcRGB, hipMemcpyHostToDevice);

        // end reading timer
        auto endReadTime = high_resolution_clock::now();

        // add elapsed iteration time
        timers["reading"].push_back(duration_cast<microseconds>(endReadTime - startReadTime).count() / 1000.0);

        // ****************************************************************** pre-processing ******************************************************************

        // start pre-process timer
        auto startPreProcessTime = high_resolution_clock::now();

        // resize frame
        rppt_resize_gpu(d_srcRGB, srcDescPtrRGB, d_dstRGB, dstDescPtrRGB, fbackImgPatchPtr, interpolationType, roiTensorPtrSrcRGB, roiType, handle1);
        // rppt_resize_host(srcRGB, srcDescPtrRGB, dstRGB, dstDescPtrRGB, fbackImgPatchPtr, interpolationType, roiTensorPtrSrcRGB, roiType, handle1);

        // -------------------- stage output dump check --------------------
        hipDeviceSynchronize();
        // hipMemcpy(dstRGB, d_dstRGB, sizeInBytesDstRGB, hipMemcpyDeviceToHost);
        // rpp_tensor_write_to_file("srcResizedRGB2", dstRGB, dstDescPtrRGB);
        // rpp_tensor_write_to_images("srcResizedRGB2", dstRGB, dstDescPtrRGB, fbackImgPatchPtr);

        // convert to gray
        rppt_color_to_greyscale_gpu(d_dstRGB, dstDescPtrRGB, d_src2, srcDescPtr, subpixelLayout, handle1);
        // rppt_color_to_greyscale_host(dstRGB, dstDescPtrRGB, src2, srcDescPtr, subpixelLayout, handle1);

        // all ops in all streams need to complete at end of pre-processing
        hipDeviceSynchronize();

        // TEMP copy cuda src buffer 2
        // hipMemcpy(d_src2, inputs + FARNEBACK_OUTPUT_FRAME_SIZE, FARNEBACK_OUTPUT_FRAME_SIZE, hipMemcpyHostToDevice);

        // -------------------- stage output dump check --------------------
        hipDeviceSynchronize();
        // hipMemcpy(src2, d_src2, sizeInBytesSrc / 2, hipMemcpyDeviceToHost);
        // rpp_tensor_write_to_file("src2", src2, srcDescPtr);
        // rpp_tensor_write_to_images("src2", src2, srcDescPtr, fbackImgPatchPtr);

        // end pre-process timer
        auto endPreProcessTime = high_resolution_clock::now();

        // add elapsed iteration time
        timers["pre-process"].push_back(duration_cast<microseconds>(endPreProcessTime - startPreProcessTime).count() / 1000.0);

        std::cerr << "\n->>>>>>>DEBUG>>>>>>>>> Finished prre-processing";

        // ****************************************************************** motion vector generation ******************************************************************

        // TO BE REMOVED Creating intermediate buffer for image outputs
        Rpp8u *d_intermU8;
        // hipMalloc(&d_intermU8, FARNEBACK_OUTPUT_FRAME_SIZE * sizeof(Rpp8u));
        Rpp32f *dh_intermF32;
        hipHostMalloc(&dh_intermF32, FARNEBACK_OUTPUT_FRAME_SIZE * 5 * sizeof(Rpp32f));

        // TEMP copy cuda resized strided outputs to dh_intermF32
        Rpp32f *dh_cudaResizdStrided;
        hipHostMalloc(&dh_cudaResizdStrided, FARNEBACK_OUTPUT_FRAME_SIZE * 2 * sizeof(Rpp32f));
        hipMemcpy(dh_cudaResizdStrided, inputsStrided, FARNEBACK_OUTPUT_FRAME_SIZE, hipMemcpyHostToHost);
        hipMemcpy(dh_cudaResizdStrided + FARNEBACK_OUTPUT_FRAME_SIZE, inputsStrided + FARNEBACK_OUTPUT_FRAME_SIZE, FARNEBACK_OUTPUT_FRAME_SIZE, hipMemcpyHostToHost);

        // start optical flow timer
        auto startOpticalFlowTime = high_resolution_clock::now();

        // create optical flow instance
        // Ptr<cuda::FarnebackOpticalFlow> ptr_calc = cuda::FarnebackOpticalFlow::create(5, 0.5, false, 15, 3, 5, 1.2, 0);
        // calculate optical flow
        hipMemset(d_motionVectorsCartesianF32Comp1, 0, FARNEBACK_OUTPUT_FRAME_SIZE * sizeof(Rpp32f));
        hipMemset(d_motionVectorsCartesianF32Comp2, 0, FARNEBACK_OUTPUT_FRAME_SIZE * sizeof(Rpp32f));
        hipDeviceSynchronize();

        rppt_farneback_optical_flow_gpu(d_src1, d_src2, d_intermU8, dh_intermF32, dh_cudaResizdStrided, srcDescPtr, d_motionVectorsCartesianF32Comp1, d_motionVectorsCartesianF32Comp2, mVecCompPlnDescPtr, 0.75f, 5, 9, 3, 5, 1.2f, handle1);
        // cv::cuda::GpuMat gpu_flow;
        // ptr_calc->calc(gpuPrevious, gpu_current, gpu_flow);


        // TO BE REMOVED Printing intermediate outputs
        hipDeviceSynchronize();

        // hipMemcpy(dh_intermF32, d_motionVectorsCartesianF32Comp1, FARNEBACK_OUTPUT_FRAME_SIZE * sizeof(Rpp32f), hipMemcpyDeviceToHost);
        // hipMemcpy(dh_intermF32 + FARNEBACK_OUTPUT_FRAME_SIZE, d_motionVectorsCartesianF32Comp2, FARNEBACK_OUTPUT_FRAME_SIZE * sizeof(Rpp32f), hipMemcpyDeviceToHost);

        // Rpp8u h_intermU8[FARNEBACK_OUTPUT_FRAME_SIZE];
        // std::cerr << "\n\nAbout to copy d to h...";
        // hipMemcpy(h_intermU8, d_intermU8, FARNEBACK_OUTPUT_FRAME_SIZE * sizeof(Rpp8u), hipMemcpyDeviceToHost);
        // std::cerr << "\n\nAbout to write to csv...";
        // rpp_tensor_write_to_file("h_intermU8", h_intermU8, srcDescPtr);
        // std::cerr << "\n\nAbout to write to jpg...";
        // rpp_tensor_write_to_images("h_intermU8", h_intermU8, srcDescPtr, fbackImgPatchPtr);

        // std::cerr << "\n\nAbout to write to csv...";
        // rpp_tensor_write_to_file("dh_intermF32_0", dh_intermF32, mVecCompPlnDescPtr);
        // rpp_tensor_write_to_file("dh_intermF32_1", dh_intermF32 + FARNEBACK_OUTPUT_FRAME_SIZE, mVecCompPlnDescPtr);
        // rpp_tensor_write_to_file("dh_intermF32_2", dh_intermF32 + (2 * FARNEBACK_OUTPUT_FRAME_SIZE), mVecCompPlnDescPtr);
        // rpp_tensor_write_to_file("dh_intermF32_3", dh_intermF32 + (3 * FARNEBACK_OUTPUT_FRAME_SIZE), mVecCompPlnDescPtr);
        // rpp_tensor_write_to_file("dh_intermF32_4", dh_intermF32 + (4 * FARNEBACK_OUTPUT_FRAME_SIZE), mVecCompPlnDescPtr);
        // std::cerr << "\n\nFarneback done...";


        hipHostFree(&dh_intermF32);
        hipHostFree(&dh_cudaResizdStrided);

        // all ops in all streams need to complete at end of motion vector generation
        hipDeviceSynchronize();

        // break;

        // end optical flow timer
        auto endOpticalFlowTime = high_resolution_clock::now();

        // add elapsed iteration time
        timers["optical flow"].push_back(duration_cast<microseconds>(endOpticalFlowTime - startOpticalFlowTime).count() / 1000.0);

        // ****************************************************************** post-processing ******************************************************************

        // Temporarily using motionVectorsCartesian from CUDA and finding min and max of their output
        // hipMemcpy(d_motionVectorsCartesianF32, motionVectorsCartesian, FARNEBACK_OUTPUT_MOTION_VECTORS_SIZE * sizeof(Rpp32f), hipMemcpyHostToDevice);
        hipMemcpy(motionVectorsCartesian, d_motionVectorsCartesianF32, FARNEBACK_OUTPUT_MOTION_VECTORS_SIZE * sizeof(Rpp32f), hipMemcpyDeviceToHost);

        // float min1 = motionVectorsCartesian[0];
        // float max1 = motionVectorsCartesian[0];
        // float min2 = motionVectorsCartesian[FARNEBACK_OUTPUT_FRAME_SIZE];
        // float max2 = motionVectorsCartesian[FARNEBACK_OUTPUT_FRAME_SIZE];

        // float *mVecPtr1Temp = motionVectorsCartesian;
        // float *mVecPtr2Temp = motionVectorsCartesian + FARNEBACK_OUTPUT_FRAME_SIZE;
        // for (int i = 0; i < FARNEBACK_OUTPUT_FRAME_SIZE; i++)
        // {
        //     min1 = std::min(min1, *mVecPtr1Temp);
        //     max1 = std::max(max1, *mVecPtr1Temp);
        //     min2 = std::min(min2, *mVecPtr2Temp);
        //     max2 = std::max(max2, *mVecPtr2Temp);
        //     mVecPtr1Temp++;
        //     mVecPtr2Temp++;
        // }

        // std::cerr << "\n\n\nMin1, Max1 = " << min1 << ", " << max1;
        // std::cerr << "\nMin2, Max2 = " << min2 << ", " << max2;

        // break;


        // -------------------- stage output dump check --------------------
        // hipDeviceSynchronize();
        // Rpp32f motionVectorsCartesianF32Comp1CPU[FARNEBACK_OUTPUT_FRAME_SIZE];
        // Rpp32f motionVectorsCartesianF32Comp2CPU[FARNEBACK_OUTPUT_FRAME_SIZE];
        // hipMemcpy(motionVectorsCartesianF32Comp1CPU, d_motionVectorsCartesianF32Comp1, FARNEBACK_OUTPUT_FRAME_SIZE * sizeof(Rpp32f), hipMemcpyDeviceToHost);
        // hipMemcpy(motionVectorsCartesianF32Comp2CPU, d_motionVectorsCartesianF32Comp2, FARNEBACK_OUTPUT_FRAME_SIZE * sizeof(Rpp32f), hipMemcpyDeviceToHost);
        // rpp_tensor_write_to_file("motionVectorsCartesianF32Comp1CPU", motionVectorsCartesianF32Comp1CPU, mVecCompPlnDescPtr);
        // rpp_tensor_write_to_file("motionVectorsCartesianF32Comp2CPU", motionVectorsCartesianF32Comp2CPU, mVecCompPlnDescPtr);

        std::cerr << "\n->>>>>>>DEBUG>>>>>>>>> Finished motion vector computations";
        // break;

        // start post-process timer
        auto startPostProcessTime = high_resolution_clock::now();

        // convert from cartesian to polar coordinates
        rppt_cartesian_to_polar_gpu(d_motionVectorsCartesianF32, mVecCartPlnGenericDescPtr, d_motionVectorsPolarF32, mVecPolrPlnGenericDescPtr, angleType, roiTensorPtrDstRGB, roiType, handle1);
        std::cerr << "\n->>>>>>>DEBUG>>>>>>>>> Finished cartesian to polar";

        // all ops in stream1 need to complete before rppt_multiply_scalar_gpu executes on stream1 and rppt_image_min_max executes on stream2
        // hipStreamSynchronize(stream1);
        hipDeviceSynchronize();

        // -------------------- stage output dump check --------------------
        // hipDeviceSynchronize();
        // Rpp32f motionVectorsPolarCPU[FARNEBACK_OUTPUT_FRAME_SIZE];
        // hipMemcpy(motionVectorsPolarCPU, d_motionVectorsPolarF32, FARNEBACK_OUTPUT_FRAME_SIZE * sizeof(Rpp32f), hipMemcpyDeviceToHost);
        // rpp_tensor_write_to_file("motionVectorsPolarCPU", motionVectorsPolarCPU, (RpptDescPtr)mVecCompPlnGenericDescPtr);
        // hipMemcpy(motionVectorsPolarCPU, d_motionVectorsPolarF32Comp1, FARNEBACK_OUTPUT_FRAME_SIZE * sizeof(Rpp32f), hipMemcpyDeviceToHost);
        // rpp_tensor_write_to_file("motionVectorsPolarCPUComp1", motionVectorsPolarCPU, (RpptDescPtr)mVecCompPlnGenericDescPtr);
        // hipMemcpy(motionVectorsPolarCPU, d_motionVectorsPolarF32Comp2, FARNEBACK_OUTPUT_FRAME_SIZE * sizeof(Rpp32f), hipMemcpyDeviceToHost);
        // rpp_tensor_write_to_file("motionVectorsPolarCPUComp2", motionVectorsPolarCPU, (RpptDescPtr)mVecCompPlnGenericDescPtr);
        // hipMemcpy(motionVectorsPolarCPU, d_motionVectorsPolarF32Comp3, FARNEBACK_OUTPUT_FRAME_SIZE * sizeof(Rpp32f), hipMemcpyDeviceToHost);
        // rpp_tensor_write_to_file("motionVectorsPolarCPUComp3", motionVectorsPolarCPU, (RpptDescPtr)mVecCompPlnGenericDescPtr);

        // normalize polar angle from 0 to 1 in hip stream1
        rppt_multiply_scalar_gpu(d_motionVectorsPolarF32Comp1, mVecCompPlnGenericDescPtr, d_motionVectorsPolarF32Comp1, mVecCompPlnGenericDescPtr, HUE_CONVERSION_FACTOR, roiTensorPtrDstRGB, roiType, handle1);
        std::cerr << "\n->>>>>>>DEBUG>>>>>>>>> Finished multiply scalar";

        // find min and max of polar magnitude in  hip stream2
        rppt_image_min_max_gpu(d_motionVectorsPolarF32, mVecCompPlnGenericDescPtr, imageMinMaxArr, imageMinMaxArrLength, roiTensorPtrDstRGB, roiType, handle1); // could be handle2
        std::cerr << "\n->>>>>>>DEBUG>>>>>>>>> Finished image min max";

        // all ops in stream2 need to complete before rppt_normalize_minmax_gpu executes on stream2
        hipDeviceSynchronize(); // could be a hipStreamSynchronize(stream2);

        std::cerr << "\n->>>>>>>DEBUG>>>>>>>>> Finished synchronnize after image min max";

        // break;

        // normalize polar magnitude from 0 to 1 in hip stream2
        rppt_normalize_minmax_gpu(d_motionVectorsPolarF32, mVecCompPlnGenericDescPtr, d_motionVectorsPolarF32Comp3, mVecCompPlnGenericDescPtr, imageMinMaxArr, imageMinMaxArrLength, 0.0f, 1.0f, roiTensorPtrDstRGB, roiType, handle1); // could be handle2
        std::cerr << "\n->>>>>>>DEBUG>>>>>>>>> Finished nnormalize";

        // all ops in all streams need to complete before rppt_hsv_to_rgbbgr_gpu executes on stream1
        // hipStreamSynchronize(stream2);
        // hipStreamSynchronize(stream1);
        hipDeviceSynchronize(); // could be a hipStreamSynchronize(stream2); followed by hipStreamSynchronize(stream1);

        // -------------------- stage output dump check --------------------
        // Rpp32f motionVectorsPolarCPU[FARNEBACK_OUTPUT_FRAME_SIZE];
        // hipMemcpy(motionVectorsPolarCPU, d_motionVectorsPolarF32Comp1, FARNEBACK_OUTPUT_FRAME_SIZE * sizeof(Rpp32f), hipMemcpyDeviceToHost);
        // rpp_tensor_write_to_file("motionVectorsPolarCPUComp1", motionVectorsPolarCPU, (RpptDescPtr)mVecCompPlnGenericDescPtr);
        // hipMemcpy(motionVectorsPolarCPU, d_motionVectorsPolarF32Comp2, FARNEBACK_OUTPUT_FRAME_SIZE * sizeof(Rpp32f), hipMemcpyDeviceToHost);
        // rpp_tensor_write_to_file("motionVectorsPolarCPUComp2", motionVectorsPolarCPU, (RpptDescPtr)mVecCompPlnGenericDescPtr);
        // hipMemcpy(motionVectorsPolarCPU, d_motionVectorsPolarF32Comp3, FARNEBACK_OUTPUT_FRAME_SIZE * sizeof(Rpp32f), hipMemcpyDeviceToHost);
        // rpp_tensor_write_to_file("motionVectorsPolarCPUComp3", motionVectorsPolarCPU, (RpptDescPtr)mVecCompPlnGenericDescPtr);

        // move resized input RGB frame from device to host in preparation for visualization
        hipMemcpy(dstInputRGB, d_dstRGB, sizeInBytesDstRGB, hipMemcpyDeviceToHost);
        std::cerr << "\n->>>>>>>DEBUG>>>>>>>>> Finished memcpy";

        // fused bitDepth + layout + colorType conversion of F32-PLN3 HSV to U8-PKD3 BGR in hip stream1
        rppt_hsv_to_rgbbgr_gpu(d_motionVectorsPolarF32Comp1, dstDescPtrHSV, d_dstRGB, dstDescPtrRGB, subpixelLayout, handle1);

        // all ops in all streams need to complete at end of post-processing
        hipDeviceSynchronize();

        // move output BGR frame from device to host in preparation for visualization
        hipMemcpy(dstRGB, d_dstRGB, sizeInBytesDstRGB, hipMemcpyDeviceToHost);
        hipDeviceSynchronize();
        std::cerr << "\n->>>>>>>DEBUG>>>>>>>>> Finished move output";

        // -------------------- stage output dump check --------------------
        // rpp_tensor_write_to_file("dstRGB", dstRGB, dstDescPtrRGB);
        // rpp_tensor_write_to_images("dstRGB", dstRGB, dstDescPtrRGB, fbackImgPatchPtr);

        // update d_src1
        hipMemcpy(d_src1, d_src2, FARNEBACK_OUTPUT_FRAME_SIZE * sizeof(Rpp8u), hipMemcpyDeviceToDevice);
        hipDeviceSynchronize();

        std::cerr << "\n->>>>>>>DEBUG>>>>>>>>> Finished update src1";

        // end post pipeline timer
        auto endPostProcessTime = high_resolution_clock::now();

        // add elapsed iteration time
        timers["post-process"].push_back(duration_cast<microseconds>(endPostProcessTime - startPostProcessTime).count() / 1000.0);

        // end full pipeline timer
        auto endFullTime = high_resolution_clock::now();

        // add elapsed iteration time
        timers["full pipeline"].push_back(duration_cast<microseconds>(endFullTime - startFullTime).count() / 1000.0);

        // ****************************************************************** visualization ******************************************************************

        // visualization
        // imshow("original", frame);
        // imshow("result", bgr);
        // rpp_tensor_write_to_images("dstInputRGB", dstInputRGB, dstDescPtrRGB, fbackImgPatchPtr);
        // rpp_tensor_write_to_images("dstRGB", dstRGB, dstDescPtrRGB, fbackImgPatchPtr);
        rpp_tensor_write_to_images_and_append_frame_to_video("dstRGB", dstRGB, dstDescPtrRGB, fbackImgPatchPtr, videoOutput, iterCount);
        std::cerr << "\n->>>>>>>DEBUG>>>>>>>>> Finished write to video";


        // break;
        // int keyboard = waitKey(1);
        // if (keyboard == 27)
        //     break;
        // hipMemcpy(src2, d_src2, sizeInBytesSrc, hipMemcpyDeviceToHost);
        // rpp_tensor_write_to_file("src2", src2, srcDescPtr);
        // rpp_tensor_write_to_images("src2", src2, srcDescPtr, fbackImgPatchPtr);
    }

    capture.release();
    videoOutput.release();
    destroyAllWindows();

    // destroy rpp handle and deallocate all buffers
    std::cerr << "\n\nAbout to free1...";
    rppDestroyGPU(handle1);
    rppDestroyGPU(handle2);
    std::cerr << "\n\nAbout to free2...";
    hipFree(&d_srcRGB);
    hipFree(&d_dstRGB);
    hipFree(&d_srcBufs);
    hipFree(&d_motionVectorsCartesianF32);
    hipFree(&d_motionVectorsPolarF32);
    std::cerr << "\n\nAbout to free3...";
    hipHostFree(&roiTensorPtrSrcRGB);
    hipHostFree(&roiTensorPtrDstRGB);
    hipHostFree(&fbackImgPatchPtr);
    hipHostFree(&imageMinMaxArr);
    std::cerr << "\n\nAbout to free4...";
    free(srcRGB);
    free(dstRGB);
    free(dstInputRGB);
    free(srcBufs);

    // display video file properties to user
    cout << "\nInput Video File - " << inputVideoFileName;
    cout << "\nFPS - " << fps;
    cout << "\nNumber of Frames - " << numOfFrames;
    cout << "\nFrame Width - " << frameWidth;
    cout << "\nFrame Height - " << frameHeight;
    cout << "\nBit Rate - " << bitRate;

    // elapsed time at each stage
    cout << "\n\nElapsed time:";
    for (auto const& timer : timers)
        cout << "\n- " << timer.first << " : " << std::accumulate(timer.second.begin(), timer.second.end(), 0.0) << " milliseconds";

    // calculate frames per second
    float opticalFlowFPS  = 1000 * (numOfFrames - 1) / std::accumulate(timers["optical flow"].begin(),  timers["optical flow"].end(),  0.0);
    float fullPipelineFPS = 1000 * (numOfFrames - 1) / std::accumulate(timers["full pipeline"].begin(), timers["full pipeline"].end(), 0.0);
    cout << "\n\nInput video FPS : " << fps;
    cout << "\nOptical flow FPS : " << opticalFlowFPS;
    cout << "\nFull pipeline FPS : " << fullPipelineFPS;
    cout << "\n";
}

int main(int argc, const char** argv)
{
    // handle inputs
    const int ARG_COUNT = 2;
    if (argc != ARG_COUNT)
    {
        printf("\nImproper Usage! Needs all arguments!\n");
        printf("\nUsage: ./Tensor_hip_optical_flow <input video file>\n");
        return -1;
    }
    string inputVideoFileName;
    inputVideoFileName = argv[1];

    // query and fix max batch size
    const auto cpuThreadCount = std::thread::hardware_concurrency();
    cout << "\n\nCPU Threads = " << cpuThreadCount;

    int device, deviceCount;
    hipGetDevice(&device);
    hipGetDeviceCount(&deviceCount);
    cout << "\nDevice = " << device;
    cout << "\nDevice Count = " << deviceCount;
    cout << "\n";

    // run optical flow
    cout << "\n\nProcessing RPP optical flow on " << inputVideoFileName << " with HIP backend...\n\n";
    rpp_optical_flow_hip(inputVideoFileName);

    return 0;
}
