#include <cuda_runtime.h>
#include <stdlib.h>

#include <iostream>
#include <string>
#include <vector>
#include <numeric>
#include <unordered_map>
#include <chrono>
#include <thread>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaoptflow.hpp>

using namespace cv;
using namespace cv::cuda;
using namespace std;
using namespace std::chrono;

void opencv_optical_flow_cuda(string inputVideoFileName)
{
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
    double fps = capture.get(CAP_PROP_FPS);
    int numOfFrames = int(capture.get(CAP_PROP_FRAME_COUNT));
    int frameWidth = int(capture.get(CAP_PROP_FRAME_WIDTH));
    int frameHeight = int(capture.get(CAP_PROP_FRAME_HEIGHT));
    int bitRate = int(capture.get(CAP_PROP_BITRATE));

    // declare gpu inputs for optical flow
    cv::cuda::GpuMat gpuFrame, gpuPrevious, gpuFlow, gpuFlowXY[2];

    // read the first frame
    cv::Mat frame, previousFrame;
    capture >> frame;

    // cpu processing on first frame - working
    // resize frame
    cv::resize(frame, frame, Size(960, 540), 0, 0, INTER_LINEAR);
    // convert to gray
    cv::cvtColor(frame, previousFrame, COLOR_BGR2GRAY);
    // upload pre-processed frame to GPU
    gpuPrevious.upload(previousFrame);

    // gpu processing on first frame - not working - core dumped
    // upload frame to GPU
    // gpuFrame.upload(frame);
    // resize frame
    // cv::cuda::resize(gpuFrame, gpuFrame, Size(960, 540), 0, 0, INTER_LINEAR);
    // convert to gray
    // cv::cuda::cvtColor(gpuFrame, gpuPrevious, COLOR_BGR2GRAY);

    // declare cpu outputs for optical flow
    cv::Mat hsv[3], angle, bgr;

    // declare gpu outputs for optical flow
    cv::cuda::GpuMat gpuMagnitude, gpuNormalizedMagnitude, gpuAngle;
    cv::cuda::GpuMat gpuHSV[3], gpuMergedHSV, gpuHSV_8u, gpuBGR;

    // set saturation to 1
    hsv[1] = cv::Mat::ones(frame.size(), CV_32F);
    gpuHSV[1].upload(hsv[1]);

    while (true)
    {
        // start full pipeline timer
        auto startFullTime = high_resolution_clock::now();

        // start reading timer
        auto startReadTime = high_resolution_clock::now();

        // capture frame-by-frame
        capture >> frame;

        if (frame.empty())
            break;

        // upload frame to GPU
        gpuFrame.upload(frame);

        // end reading timer
        auto endReadTime = high_resolution_clock::now();

        // add elapsed iteration time
        timers["reading"].push_back(duration_cast<microseconds>(endReadTime - startReadTime).count() / 1000.0);

        // start pre-process timer
        auto startPreProcessTime = high_resolution_clock::now();

        // resize frame
        cv::cuda::resize(gpuFrame, gpuFrame, Size(960, 540), 0, 0, INTER_LINEAR);

        // convert to gray
        cv::cuda::GpuMat gpuCurrent;
        cv::cuda::cvtColor(gpuFrame, gpuCurrent, COLOR_BGR2GRAY);

        // end pre-process timer
        auto endPreProcessTime = high_resolution_clock::now();

        // add elapsed iteration time
        timers["pre-process"].push_back(duration_cast<microseconds>(endPreProcessTime - startPreProcessTime).count() / 1000.0);

        // start optical flow timer
        auto startOpticalFlowTime = high_resolution_clock::now();

        // create optical flow instance
        Ptr<cuda::FarnebackOpticalFlow> ptr_calc = cuda::FarnebackOpticalFlow::create(5, 0.5, false, 15, 3, 5, 1.2, 0);
        // calculate optical flow
        ptr_calc->calc(gpuPrevious, gpuCurrent, gpuFlow);

        // end optical flow timer
        auto endOpticalFlowTime = high_resolution_clock::now();

        // add elapsed iteration time
        timers["optical flow"].push_back(duration_cast<microseconds>(endOpticalFlowTime - startOpticalFlowTime).count() / 1000.0);

        // start post-process timer
        auto startPostProcessTime = high_resolution_clock::now();

        // -------------------- stage output dump check --------------------
        // cv::Mat flow;
        // gpuFlow.download(flow);
        // std::cout << "\ngpuFlow:\n" << flow << "\n";

        // split the output flow into 2 vectors
        cv::cuda::split(gpuFlow, gpuFlowXY);

        // -------------------- stage output dump check --------------------
        // cv::Mat flowXY[2];
        // gpuFlowXY[0].download(flowXY[0]);
        // gpuFlowXY[1].download(flowXY[1]);
        // std::cout << "\nflowXY[0]:\n" << flowXY[0] << "\n";
        // std::cout << "\nflowXY[1]:\n" << flowXY[1] << "\n";

        // convert from cartesian to polar coordinates
        cv::cuda::cartToPolar(gpuFlowXY[0], gpuFlowXY[1], gpuMagnitude, gpuAngle, true);

        // -------------------- stage output dump check --------------------
        // cv::Mat magnitude_temp, angle_temp;
        // gpuMagnitude.download(magnitude_temp);
        // gpuAngle.download(angle_temp);
        // std::cout << "\ngpuMagnitude:\n" << magnitude_temp << "\n";
        // std::cout << "\ngpuAngle:\n" << angle_temp << "\n";

        // normalize magnitude from 0 to 1
        cv::cuda::normalize(gpuMagnitude, gpuNormalizedMagnitude, 0.0, 1.0, NORM_MINMAX, -1);

        // -------------------- stage output dump check --------------------
        // cv::Mat normalizedMagnitude_temp;
        // gpuNormalizedMagnitude.download(normalizedMagnitude_temp);
        // std::cout << "\ngpuNormalizedMagnitude:\n" << normalizedMagnitude_temp << "\n";

        // get angle of optical flow
        gpuAngle.download(angle);
        angle *= ((1 / 360.0) * (180 / 255.0));

        // -------------------- stage output dump check --------------------
        // std::cout << "\nangle:\n" << angle << "\n";

        // build hsv image
        gpuHSV[0].upload(angle);
        gpuHSV[2] = gpuNormalizedMagnitude;
        cv::cuda::merge(gpuHSV, 3, gpuMergedHSV);

        // -------------------- stage output dump check --------------------
        // cv::Mat hsv_temp0, hsv_temp1, hsv_temp2, hsvMerged_temp;
        // gpuHSV[0].download(hsv_temp0);
        // gpuHSV[1].download(hsv_temp1);
        // gpuHSV[2].download(hsv_temp2);
        // gpuMergedHSV.download(hsvMerged_temp);
        // std::cout << "\ngpuHSV[0]:\n" << hsv_temp0 << "\n";
        // std::cout << "\ngpuHSV[1]:\n" << hsv_temp1 << "\n";
        // std::cout << "\ngpuHSV[2]:\n" << hsv_temp2 << "\n";
        // std::cout << "\ngpuMergedHSV:\n" << hsvMerged_temp << "\n";

        // multiply each pixel value to 255
        gpuMergedHSV.cv::cuda::GpuMat::convertTo(gpuHSV_8u, CV_8U, 255.0);

        // convert hsv to bgr
        cv::cuda::cvtColor(gpuHSV_8u, gpuBGR, COLOR_HSV2BGR);

        // send original frame from GPU back to CPU
        gpuFrame.download(frame);

        // send result from GPU back to CPU
        gpuBGR.download(bgr);

        // -------------------- stage output dump check --------------------
        // std::cout << "\nframe:\n" << frame << "\n";
        // std::cout << "\nbgr:\n" << bgr << "\n";

        // update previousFrame value
        gpuPrevious = gpuCurrent;

        // end post pipeline timer
        auto endPostProcessTime = high_resolution_clock::now();

        // add elapsed iteration time
        timers["post-process"].push_back(duration_cast<microseconds>(endPostProcessTime - startPostProcessTime).count() / 1000.0);

        // end full pipeline timer
        auto endFullTime = high_resolution_clock::now();

        // add elapsed iteration time
        timers["full pipeline"].push_back(duration_cast<microseconds>(endFullTime - startFullTime).count() / 1000.0);

        // visualization
        // imshow("original", frame);
        // imshow("result", bgr);
        cv::imwrite("result.png", bgr);
        int keyboard = waitKey(1);
        if (keyboard == 27)
            break;

        break;
    }

    capture.release();
    destroyAllWindows();

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
        printf("\nUsage: ./cuda_optical_flow <input video file>\n");
        return -1;
    }
    string inputVideoFileName;
    inputVideoFileName = argv[1];

    // query and fix max batch size
    const auto cpuThreadCount = std::thread::hardware_concurrency();
    cout << "\n\nCPU threads = " << cpuThreadCount;

    int device, deviceCount;
    cudaGetDevice(&device);
    cudaGetDeviceCount(&deviceCount);
    cout << "\nDevice = " << device;
    cout << "\nDevice Count = " << deviceCount;
    cout << "\n";

    // run optical flow
    cout << "\n\nProcessing OpenCV optical flow on " << inputVideoFileName << " with CUDA backend...\n\n";
    opencv_optical_flow_cuda(inputVideoFileName);

    return 0;
}
