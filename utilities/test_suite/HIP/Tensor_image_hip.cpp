/*
MIT License

Copyright (c) 2019 - 2025 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include <stdio.h>
#include <dirent.h>
#include <string.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include "rpp.h"
#include "../rpp_test_suite_image.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <time.h>
#include <omp.h>
#include <fstream>

using namespace cv;
using namespace std;


// Loads all valid images from a directory into a vector.
vector<Mat> loadBatchImages(const string& directory, int& noOfImages, bool isColor) {
    vector<Mat> images;
    DIR* dir;
    struct dirent* entry;

    // Try opening the directory
    if ((dir = opendir(directory.c_str())) == NULL) {
        cerr << "Could not open directory: " << directory << endl;
        return images;
    }

    // Read all entries in the directory
    while ((entry = readdir(dir)) != NULL) {
        string filename = entry->d_name;

        // Skip "." and ".."
        if (filename == "." || filename == "..") continue;

        // Check file extension for common image formats
        string ext = filename.substr(filename.find_last_of(".") + 1);
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext != "jpg" && ext != "jpeg" && ext != "png" && ext != "bmp" && ext != "tiff")
            continue;

        // Build full file path and load image
        string filePath = directory + "/" + filename;
        Mat img = imread(filePath, isColor ? IMREAD_COLOR : IMREAD_GRAYSCALE);
        images.push_back(img);
    }

    closedir(dir);                  // Close directory stream
    noOfImages = images.size();     // Set output batch size
    return images;                  // Return the list of loaded images
}

// Helper function to initialize descriptors and ROI
void initializeDescriptorsAndRoi(const vector<Mat>& imgs, vector<RpptDesc>& srcDescs, vector<RpptDesc>& dstDescs, int offsetInBytes, vector<RpptROI>& rois)
{
    int channels = imgs[0].channels();
    int batchSize = imgs.size();

    for (int i = 0; i < batchSize; ++i)
    {
        const Mat& img = imgs[i];

        // Set ROI
        rois[i].xywhROI.xy.x = 0;
        rois[i].xywhROI.xy.y = 0;
        rois[i].xywhROI.roiWidth = img.cols;
        rois[i].xywhROI.roiHeight = img.rows;

        // Set descriptor dimensions
        srcDescs[i].h = dstDescs[i].h = img.rows;
        srcDescs[i].w = dstDescs[i].w = img.cols;
        srcDescs[i].offsetInBytes = offsetInBytes;
        srcDescs[i].c = dstDescs[i].c = channels;
        srcDescs[i].n = dstDescs[i].n = 1;
        srcDescs[i].dataType = dstDescs[i].dataType = RpptDataType::U8;
        srcDescs[i].strides.nStride = dstDescs[i].strides.nStride = img.rows * img.cols * channels;

        if (channels == 3)
        {
            // NHWC layout
            srcDescs[i].strides.hStride = dstDescs[i].strides.hStride = img.cols * channels;
            srcDescs[i].strides.wStride = dstDescs[i].strides.wStride = channels;
            srcDescs[i].strides.cStride = dstDescs[i].strides.cStride = 1;
            srcDescs[i].layout = dstDescs[i].layout = RpptLayout::NHWC;
        }
        else
        {
            // NCHW layout
            srcDescs[i].strides.hStride = dstDescs[i].strides.hStride = img.cols;
            srcDescs[i].strides.wStride = dstDescs[i].strides.wStride = 1;
            srcDescs[i].strides.cStride = dstDescs[i].strides.cStride = img.cols * img.rows;
            srcDescs[i].layout = dstDescs[i].layout = RpptLayout::NCHW;
        }
    }
}

// sets descriptor data types of src/dst
inline void set_descriptor_data_type_name(int BitDepthTestMode, string &funcName)
{
    if (BitDepthTestMode == U8_TO_U8)
        funcName += "_u8_";
    else if (BitDepthTestMode == F16_TO_F16)
        funcName += "_f16_";
    else if (BitDepthTestMode == F32_TO_F32)
        funcName += "_f32_";
    else if (BitDepthTestMode == U8_TO_F16)
        funcName += "_u8_f16_";
    else if (BitDepthTestMode == U8_TO_F32)
        funcName += "_u8_f32_";
    else if (BitDepthTestMode == I8_TO_I8)
        funcName += "_i8_";
    else if (BitDepthTestMode == U8_TO_I8)
        funcName += "_u8_i8_";
}

int main(int argc, char **argv)
{
    // Handle inputs
    const int MIN_ARG_COUNT = 19;

    char *src = argv[1];
    char *srcSecond = argv[2];
    string dst = argv[3];

    int BitDepthTestMode = atoi(argv[4]);
    unsigned int outputFormatToggle = atoi(argv[5]);
    int testCase = atoi(argv[6]);
    int numRuns = atoi(argv[8]);
    int testType = atoi(argv[9]);     // 0 for unit and 1 for performance test
    int layoutType = atoi(argv[10]); // 0 for pkd3 / 1 for pln3 / 2 for pln1
    int qaFlag = atoi(argv[12]);
    int decoderType = atoi(argv[13]);
    int batchSize = atoi(argv[14]);

    bool additionalParamCase = (additionalParamCases.find(testCase) != additionalParamCases.end());
    bool kernelSizeCase = (kernelSizeCases.find(testCase) != kernelSizeCases.end());
    bool dualInputCase = (dualInputCases.find(testCase) != dualInputCases.end());
    bool randomOutputCase = (randomOutputCases.find(testCase) != randomOutputCases.end());
    bool nonQACase = (nonQACases.find(testCase) != nonQACases.end());
    bool interpolationTypeCase = (interpolationTypeCases.find(testCase) != interpolationTypeCases.end());
    bool reductionTypeCase = (reductionTypeCases.find(testCase) != reductionTypeCases.end());
    bool noiseTypeCase = (noiseTypeCases.find(testCase) != noiseTypeCases.end());
    bool pln1OutTypeCase = (pln1OutTypeCases.find(testCase) != pln1OutTypeCases.end());

    unsigned int verbosity = atoi(argv[11]);
    unsigned int additionalParam = additionalParamCase ? atoi(argv[7]) : 1;
    int roiList[4] = {atoi(argv[15]), atoi(argv[16]), atoi(argv[17]), atoi(argv[18])};
    string scriptPath = argv[19];

    if (verbosity == 1)
    {
        cout << "\nInputs for this test case are:";
        cout << "\nsrc1 = " << argv[1];
        cout << "\nsrc2 = " << argv[2];
        if (testType == UNIT_TEST) // unit test mode
            cout << "\ndst = " << argv[3];
        cout << "\nu8 / f16 / f32 / u8->f16 / u8->f32 / i8 / u8->i8 (0/1/2/3/4/5/6) = " << argv[4];
        cout << "\noutputFormatToggle (pkd->pkd = 0 / pkd->pln = 1) = " << argv[5];
        cout << "\ncase number (0:91) = " << argv[6];
        cout << "\nnumber of times to run = " << argv[8];
        cout << "\ntest type - (0 = unit tests / 1 = performance tests) = " << argv[9];
        cout << "\nlayout type - (0 = PKD3/ 1 = PLN3/ 2 = PLN1) = " << argv[10];
        cout << "\nqa mode - 0/1 = " << argv[12];
        cout << "\ndecoder type - (0 = TurboJPEG / 1 = OpenCV) = " << argv[13];
        cout << "\nbatch size = " << argv[14];
    }

    if (argc < MIN_ARG_COUNT)
    {
        cout << "\nImproper Usage! Needs all arguments!\n";
        cout << "\nUsage: <src1 folder> <src2 folder (place same as src1 folder for single image functionalities)> <dst folder> <u8 = 0 / f16 = 1 / f32 = 2 / u8->f16 = 3 / u8->f32 = 4 / i8 = 5 / u8->i8 = 6> <outputFormatToggle (pkd->pkd = 0 / pkd->pln = 1)> <case number = 0:87> <number of runs > 0> <layout type (0 = PKD3/ 1 = PLN3/ 2 = PLN1)> <qa mode (0/1)> <decoder type (0/1)> <batch size > 1> <roiList> <verbosity = 0/1>>\n";
        return -1;
    }

    if (layoutType == 2)
    {
        if(testCase == COLOR_TWIST || testCase == COLOR_CAST || testCase == GLITCH || testCase == COLOR_TEMPERATURE || testCase == COLOR_TO_GREYSCALE || testCase == HUE || testCase == SATURATION)
        {
            cout << "\ncase " << testCase << " does not exist for PLN1 layout\n";
            return RPP_ERROR_NOT_IMPLEMENTED;
        }
        else if (outputFormatToggle != 0)
        {
            cout << "\nPLN1 cases don't have outputFormatToggle! Please input outputFormatToggle = 0\n";
            return RPP_ERROR_NOT_IMPLEMENTED;
        }
    }

    if(pln1OutTypeCase && outputFormatToggle != 0)
    {
        cout << "\ntest case " << testCase << " don't have outputFormatToggle! Please input outputFormatToggle = 0\n";
        return RPP_ERROR_NOT_IMPLEMENTED;
    }
    else if(batchSize > MAX_BATCH_SIZE)
    {
        std::cerr << "\n Batchsize should be less than or equal to "<< MAX_BATCH_SIZE << " Aborting!";
        exit(0);
    }
    else if(testCase == RICAP && batchSize < 2)
    {
        std::cerr<<"\n RICAP only works with BatchSize > 1";
        exit(0);
    }

    // Get function name
    string funcName = augmentationMap[testCase];
    if (funcName.empty())
    {
        if (testType == UNIT_TEST) // unit test mode
            cout << "\ncase " << testCase << " is not supported\n";

        return -1;
    }

    // Determine the type of function to be used based on the specified layout type
    string funcType = set_function_type(layoutType, pln1OutTypeCase, outputFormatToggle, "HIP");

    // String ops on input path
    string inputPath = src;
    inputPath += "/";

    // Set src/dst data types in tensor descriptors
    string func = funcName;
    set_descriptor_data_type_name(BitDepthTestMode, func);
    func += funcType;

    if (kernelSizeCase)
    {
        func += "_kernelSize";
        func += std::to_string(additionalParam);
    }
    if(!qaFlag)
    {
        dst += "/";
        dst += func;
    }
    Rpp32s additionalStride = 0;
    if (kernelSizeCase)
        additionalStride = additionalParam / 2;
    Rpp32u srcOffsetInBytes = 0;
    srcOffsetInBytes = (kernelSizeCase) ? (12 * (additionalParam / 2)) : 0;
    int noOfImages = 0, missingFuncFlag = 0, i;
    bool isColor = (layoutType == 2) ? false : true;
    vector<Mat> inputVec = loadBatchImages(src, noOfImages, isColor);

    if (noOfImages < batchSize)
    {
        if (noOfImages == 0) { cerr << "No images found!"; return -1; }
        
        for (int i = noOfImages; i < batchSize; i++)
            inputVec.push_back(inputVec[noOfImages - 1]);
        noOfImages = batchSize; // Update count to match requested batch size
    }

    vector<Mat> outputVec(noOfImages);
    for (int i = 0; i < noOfImages; ++i)
        outputVec[i] = Mat(inputVec[i].rows, inputVec[i].cols, inputVec[i].type());
    vector<RpptDesc> srcDescPtr(noOfImages), dstDescPtr(noOfImages);
    vector<RpptROI> roi(noOfImages);
    RpptImageBorderType borderType = RpptImageBorderType::REPLICATE;

    // Run case-wise RPP API and measure time
    rppHandle_t handle;
    hipStream_t stream;
    CHECK_RETURN_STATUS(hipStreamCreate(&stream));
    RppBackend backend = RppBackend::RPP_HIP_BACKEND;
    rppCreate(&handle, 1, 0, stream, backend);
    double maxWallTime = 0, minWallTime = 500, avgWallTime = 0;
    double wallTime;
    string testCaseName;
    initializeDescriptorsAndRoi(inputVec, srcDescPtr, dstDescPtr, srcOffsetInBytes, roi);

    // case-wise RPP API and measure time script for Unit and Performance test
    cout << "\nRunning " << func << " " << numRuns << " times (each time with a batch size of " << batchSize << " images) and computing mean statistics...";
    for (int perfRunCount = 0; perfRunCount < numRuns; perfRunCount++)
    {
        for (int i = 0; i < noOfImages; i++)
        {
            RppStatus errorCodeCapture = RPP_SUCCESS;
            double startWallTime, endWallTime;

            void *d_input, *d_output;
            size_t inputSize = inputVec[i].rows * inputVec[i].cols * inputVec[i].channels() * sizeof(Rpp8u);
            size_t outputSize = outputVec[i].rows * outputVec[i].cols * outputVec[i].channels() * sizeof(Rpp8u);
            CHECK_RETURN_STATUS(hipMalloc(&d_input, inputSize));
            CHECK_RETURN_STATUS(hipMalloc(&d_output, outputSize));
            CHECK_RETURN_STATUS(hipMemcpy(d_input, inputVec[i].data, inputSize, hipMemcpyHostToDevice));

            switch (testCase)
            {
                case BRIGHTNESS:
                {
                    testCaseName = "brightness";
                    Rpp32f alpha = 1.75f;
                    Rpp32f beta = 50.0f;

                    startWallTime = omp_get_wtime();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 || BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_brightness_gpu(d_input, &srcDescPtr[i], d_output, &dstDescPtr[i], &alpha, &beta, &roi[i], RpptRoiType::XYWH, handle);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case BOX_FILTER:
                {
                    testCaseName = "box_filter";
                    Rpp32u kernelSize = additionalParam;

                    if (borderType != RpptImageBorderType::REPLICATE)
                    {
                        missingFuncFlag = 1;
                        break;
                    }

                    startWallTime = omp_get_wtime();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 || BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_box_filter_gpu(d_input, &srcDescPtr[i], d_output, &dstDescPtr[i], kernelSize, borderType, &roi[i], RpptRoiType::XYWH, handle);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                default:
                {
                    missingFuncFlag = 1;
                    break;
                }
            }

            CHECK_RETURN_STATUS(hipDeviceSynchronize());
            endWallTime = omp_get_wtime();
            CHECK_RETURN_STATUS(hipMemcpy(outputVec[i].data, d_output, outputSize, hipMemcpyDeviceToHost));
            CHECK_RETURN_STATUS(hipFree(d_input));
            CHECK_RETURN_STATUS(hipFree(d_output));

            if (missingFuncFlag == 1)
            {
                cout << "\nThe functionality " << func << " doesn't yet exist in RPP\n";
                return RPP_ERROR_NOT_IMPLEMENTED;
            }
            if (errorCodeCapture != RPP_SUCCESS)
            {
                cout << "\nThe functionality " << func << " returned an error status " << rppStatusToString[errorCodeCapture] << " on run number " << perfRunCount + 1 << " of " << numRuns << " runs.\n";
                return errorCodeCapture;
            }

            wallTime = endWallTime - startWallTime;
            maxWallTime = max(maxWallTime, wallTime);
            minWallTime = min(minWallTime, wallTime);
            avgWallTime += wallTime;
        }
    }

    if (testType == UNIT_TEST) // unit test mode
    {
        cout <<"\n\n";
        cout << "GPU Backend Wall Time: " << avgWallTime * 1000 / (numRuns * noOfImages) <<" ms/image";
         // Ensure destination folder exists
        mkdir(dst.c_str(), 0700);

        for(int i = 0; i < noOfImages; i++)
        {
            string separator = (dst.back() == '/') ? "" : "/";
            string currentFileName = dst + separator + to_string(i) + ".jpg";
                
            Mat saveImg = outputVec[i];
            
            if (saveImg.empty()) {
                cerr << "\n[Error] Output image " << i << " is empty!";
                continue;
            }

            imwrite(currentFileName, saveImg);

            cout << "\nSaved: " << currentFileName;
        }
    }
    rppDestroy(handle, backend);
    if(testType == PERFORMANCE_TEST) // performance test mode
    {
        // Display measured times
        maxWallTime *= 1000;
        minWallTime *= 1000;
        avgWallTime *= 1000;
        avgWallTime /= (numRuns);
        cout << fixed << "\n Running : "<< func << "\nmax,min,avg wall times in ms/batch = " << maxWallTime << "," << minWallTime << "," << avgWallTime;
    }
    
    return 0;
}
