//
// Created by 单淳劼 on 2022/7/14.
//

#ifndef INC_22_07_14_TENSORRT_CPP_DEMO_INFER_H
#define INC_22_07_14_TENSORRT_CPP_DEMO_INFER_H

#include "cuda_runtime_api.h"
#include "NvInfer.h"
#include "logging.h"
#include "logger.h"
#include <fstream>
#include <iostream>
#include <numeric>
#include <cassert>
#include <chrono>
#include <opencv2/opencv.hpp>

using namespace nvinfer1;
//using namespace sample;

class KeyframeInfer {
public:
    KeyframeInfer(std::string& trtEnginePath, cv::Size& inputSize, int singleVideoFrameCount, std::vector<float> Mean);

    ~KeyframeInfer();

    std::vector<int> doInference(std::vector<cv::Mat>& sVideoPath, int sampleStride);

private:
    std::string mEnginePath;
    logger::Logger mGLogger;
    IRuntime* mRuntime;
    ICudaEngine* mEngine;
    IExecutionContext* mContext;
    cudaStream_t mStream;

    cv::Size mcvInputSize;
    int mSingleVideoFrameCount;

    int mInputSize;
    int mOutputSize;
    int mInputIndex;
    int mOutputIndex;

    float* mBlob;
    float* mProb;
    void* mBuffers[2];

    std::vector<float> mMean;

    int initialize();

    std::vector<cv::Mat> videoInput(std::vector<cv::Mat> frames, int sampleStride, std::vector<std::vector<int>>& vVideosIdx);

    int genVideosIdx(int totalFrameCount, int sampleStride, std::vector<std::vector<int>>& vVideosIdx) const;

//    void videoIdxPostProcess(std::vector<std::vector<int>>& vVideoIdx);

    std::vector<float> doSingleInfer(std::vector<cv::Mat>& vSingleVideo);

    int doInference(std::vector<cv::Mat>& vFrames,
                    std::vector<std::vector<int>>& vVideosIdx,
                    std::vector<float>& vMeans);

    void blobFromVideo(std::vector<cv::Mat>& vSingleVideo);

//    int doInference(std::vector<std::vector<cv::Mat>>& vMatVideos, std::vector<std::vector<int>>& vTotalIdxMax, std::vector<std::vector<int>>& vTotalIdxMin);

    static std::vector<float> decodeOutputs(int totalFrameCount, std::vector<std::vector<float>>& vProbs, std::vector<std::vector<int>>& vVideosIdx);

    void decodeSingleOutput(std::vector<float>& vProb, std::vector<int>& vIdxMax, std::vector<int>& vIdxMin);

    static float inline decodeSingleOutput(std::vector<float>& vProb)
    {
        return std::accumulate(vProb.begin(), vProb.end(), 0.0f) / static_cast<float>(vProb.size());
    }

    void findPeaks(std::vector<float>& vProb, std::vector<int>& vIdxMax, std::vector<int>& vIdxMin);

	std::vector<int> findPeak(std::vector<float> num, int count);

    static void checkStatus(cudaError status)
    {
        do
        {
            auto ret = (status);
            if (ret != 0)
            {
                std::cerr << "Cuda failure: " << ret << std::endl;
                abort();
            }
        } while (0);
    }

};


#endif //INC_22_07_14_TENSORRT_CPP_DEMO_INFER_H
