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
vector<Mat> loadBatchImages(const string& directory, int& batchSize, bool isColor) {
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
    batchSize = images.size();      // Set output batch size
    return images;                  // Return the list of loaded images
}


// Helper function to initialize descriptors and ROI
void initializeDescriptorsAndRoi(const vector<Mat>& imgs, vector<RpptDesc>& srcDescs, vector<RpptDesc>& dstDescs, vector<RpptROI>& rois)
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
        cout << "\nlayout type - (0 = PKD3 / 1 = PLN3 / 2 = PLN1) = " << argv[10];
        cout << "\nqa mode - 0/1 = " << argv[12];
        cout << "\ndecoder type - (0 = TurboJPEG / 1 = OpenCV) = " << argv[13];
        cout << "\nbatch size = " << argv[14];
    }

    if (argc < MIN_ARG_COUNT)
    {
        cout << "\nImproper Usage! Needs all arguments!\n";
        cout << "\nUsage: <src1 folder> <src2 folder (place same as src1 folder for single image functionalities)> <dst folder> <u8 = 0 / f16 = 1 / f32 = 2 / u8->f16 = 3 / u8->f32 = 4 / i8 = 5 / u8->i8 = 6> <outputFormatToggle (pkd->pkd = 0 / pkd->pln = 1)> <case number = 0:87> <number of runs > 0> <layout type (layout type - (0 = PKD3 / 1 = PLN3 / 2 = PLN1)> < qa mode (0/1)> <decoder type (0/1)> <batch size > 1> <roiList> <verbosity = 0/1>>\n";
    }

    if (layoutType == 2)
    {
        if(testCase == COLOR_CAST || testCase == GLITCH || testCase == COLOR_TWIST || testCase == COLOR_TEMPERATURE || testCase == COLOR_TO_GREYSCALE || testCase == HUE || testCase == SATURATION)
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
    else if (reductionTypeCase && outputFormatToggle != 0)
    {
        cout << "\nReduction Kernels don't have outputFormatToggle! Please input outputFormatToggle = 0\n";
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

    if(batchSize > MAX_BATCH_SIZE)
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
        if (testType == UNIT_TEST)  // unit test mode
            cout << "\ncase " << testCase << " is not supported\n";

        return -1;
    }
    string funcType = set_function_type(layoutType, pln1OutTypeCase, outputFormatToggle, "HOST");

    string inputPath = src;
    inputPath += "/";
    string func = funcName;
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
    int noOfImages = 0, missingFuncFlag = 0;
    bool isColor = (layoutType == 2) ? false : true;
    vector<Mat> inputVec = loadBatchImages(src, noOfImages, isColor);

    if (noOfImages < batchSize)
    {
        if (noOfImages == 0) { cerr << "No images found!"; return -1; }
        
        for (int i = noOfImages; i < batchSize; i++)
            inputVec.push_back(inputVec[noOfImages - 1]);
        noOfImages = batchSize; // Update count to match requested batch size
    }
    // If directory had MORE images than batchSize, resize down
    else if (noOfImages > batchSize)
    {
        inputVec.resize(batchSize);
        noOfImages = batchSize;
    }

    vector<Mat> outputVec(noOfImages);
    for (int i = 0; i < noOfImages; ++i)
        outputVec[i] = Mat(inputVec[i].rows, inputVec[i].cols, inputVec[i].type());
    vector<RpptDesc> srcDescPtr(noOfImages), dstDescPtr(noOfImages);
    vector<RpptROI> roi(noOfImages);
    RpptImageBorderType borderType = RpptImageBorderType::REPLICATE;
    // Set the number of threads to be used by OpenMP pragma for RPP batch processing on host.
    // If numThreads value passed is 0, number of OpenMP threads used by RPP will be set to batch size
    Rpp32u numThreads = 0;
    rppHandle_t handle;
    RppBackend backend = RppBackend::RPP_HOST_BACKEND;
    rppCreate(&handle, 1, numThreads, nullptr, backend);
    double maxWallTime = 0, minWallTime = 500, avgWallTime = 0;
    double cpuTime, wallTime;
    string testCaseName;
    initializeDescriptorsAndRoi(inputVec, srcDescPtr, dstDescPtr, roi);

    for (int perfRunCount = 0; perfRunCount < numRuns; perfRunCount++)
    {
        for(int i = 0; i < noOfImages; i++)
        {
            RppStatus errorCodeCapture = RPP_SUCCESS;
            clock_t startCpuTime, endCpuTime;
            double startWallTime, endWallTime;
            switch (testCase)
            {
                case BRIGHTNESS:
                {
                    testCaseName = "brightness";
                    Rpp32f alpha = 1.75f;
                    Rpp32f beta = 50.0f;

                    startWallTime = omp_get_wtime();
                    startCpuTime = clock();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 || BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_brightness_host(inputVec[i].data, &srcDescPtr[i], outputVec[i].data, &dstDescPtr[i], alpha, beta, &roi[i], RpptRoiType::XYWH, handle);
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
                    startCpuTime = clock();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 || BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_box_filter_host(inputVec[i].data, &srcDescPtr[i], outputVec[i].data, &dstDescPtr[i], kernelSize, borderType, &roi[i], RpptRoiType::XYWH, handle);
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
            endCpuTime = clock();
            endWallTime = omp_get_wtime();
            cpuTime = ((double)(endCpuTime - startCpuTime)) / CLOCKS_PER_SEC;
            wallTime = endWallTime - startWallTime;

            if (missingFuncFlag == 1)
            {
                cout << "\nThe functionality " << " doesn't yet exist in RPP\n";
                return RPP_ERROR_NOT_IMPLEMENTED;
            }
            maxWallTime = std::max(maxWallTime, wallTime);
            minWallTime = std::min(minWallTime, wallTime);
            avgWallTime += wallTime;
        }
    }

    cpuTime *= 1000;
    wallTime *= 1000;

    if (testType == UNIT_TEST) 
    {
        cout <<"\n\n";
        cout <<"CPU Backend Clock Time: "<< cpuTime <<" ms/batch"<< endl;
        cout <<"CPU Backend Wall Time: "<< wallTime <<" ms/batch";
        
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

    if(testType == PERFORMANCE_TEST)
    {
        maxWallTime *= 1000;
        minWallTime *= 1000;
        avgWallTime *= 1000;
        avgWallTime /= (numRuns * noOfImages);
        cout << fixed << "\nmax,min,avg wall times in ms/batch = " << maxWallTime << "," << minWallTime << "," << avgWallTime;
    }

    cout << endl;

    return 0;
}
