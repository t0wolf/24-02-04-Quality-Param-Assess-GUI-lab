//
// Created by Ziyon on 2022/7/12.
//
#pragma once
//#include "argsParser.h"
//#include "buffers.h"
//#include "common.h"
#include "logger.h"
//#include "parserOnnxConfig.h"
#include "NvInfer.h"
#include <cuda_runtime_api.h>

//#include <opencv.hpp>
#include "opencv2/opencv.hpp"
#include "logging.h"
#include "cplusfuncs.h"
#include "typemappings.h"
#include "general_utils.h"



//using namespace sample;
//using namespace samplesCommon;
using namespace nvinfer1;
//using namespace nvonnxparser;

class GainInfer
{
public:

	GainInfer(std::string& trtEnginePath, cv::Size& inputSize, std::vector<float> mMean, std::vector<float> mStd);

	~GainInfer();

	void doInference(std::vector<cv::Mat> video, std::vector<int>& frame_gain_scores); // 预测整个视频

private:

	void build();

	std::vector<cv::Mat> videoInput(std::vector<cv::Mat> video);

	std::vector<int> Inference(std::vector<cv::Mat>& vFrames);

	std::vector<float> doSingleInfer(cv::Mat& image);

	void blobFromImage(cv::Mat& image);

	void preprocess(cv::Mat& src, cv::Mat& dst);

	std::pair<int, float> argmax(std::vector<float>& vProb);

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
	std::vector<float> mMean;
	std::vector<float> mStd;
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