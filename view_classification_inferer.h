#ifndef VIDEOCLASSIFICATIONINFERER_H
#define VIDEOCLASSIFICATIONINFERER_H

#include <iostream>
#include <fstream>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv.hpp>
#include "logger.h"
#include "typemapping.h"

using namespace nvinfer1;

class ViewClassificationInferer
{
public:
    ViewClassificationInferer(std::string trtEnginePath, cv::Size inputSize, int singleVideoFrameCount = 64, int inferenceTime = 7);

    view_fidxs doInference(std::vector<cv::Mat> video);

private:
    int initialize();

private:
    // nvinfer1::IExecutionContext* m_context;

    std::vector<cv::Mat> videoInput(std::vector<cv::Mat> video, std::vector<std::vector<int>>& vVideosIdx);

    int genVideosIdx(int totalFrameCount, std::vector<int> randnum, std::vector<std::vector<int>>& vVideosIdx) const;

    view_fidxs Inference(std::vector<cv::Mat>& vFrames,std::vector<std::vector<int>>& vVideosIdx);

    std::vector<float> doSingleInfer(std::vector<cv::Mat>& vSingleVideo);

    void blobFromVideo(std::vector<cv::Mat>& vSingleVideo);

    view_fidxs fidxsPostprocess(view_fidxs view_frameidxs, int video_size);

    logger::Logger mGLogger;
    IRuntime* mRuntime;
    ICudaEngine* mEngine;
    IExecutionContext* mContext;
    cudaStream_t mStream;
    std::string mEnginePath;
    cv::Size mcvInputSize;
    int mSingleVideoFrameCount;
    int mInputSize;
    int mOutputSize;
    int mInputIndex;
    int mOutputIndex;
    int mClassNum;
    int mFrameNum;
    int mInferenceTime;
    std::vector<float> mMean = {114.7748, 107.7354, 99.475};;
    //std::shared_ptr<ICudaEngine> engine;
    //SampleUniquePtr<IExecutionContext> context;
    float* mBlob;
    float* mProb;
    void* mBuffers[2];

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

#endif // VIDEOCLASSIFICATIONINFERER_H
