#pragma once
#include <iostream>
#include <fstream>
#include <cassert>
#include <chrono>
#include <opencv.hpp>
#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include "logger.h"
#include <QDebug>

class IntegrityClassification
{
public:
	IntegrityClassification(std::string& enginePath);
	~IntegrityClassification();
	void initialize();
	int inference(std::vector<cv::Mat>& frames);

private:
	void sampleFrame(int totalFrame, std::vector<std::vector<int>>& vInds);
	void sampleFrame(std::vector<cv::Mat>& vMats);
	void videoInput(cv::VideoCapture& cap, std::vector<cv::Mat>& vMats);
	std::pair<int, float> singleInfer(std::vector<cv::Mat>& vSingleMats);
	void preprocess(cv::Mat& mat);
	void blobFromVideo(std::vector<cv::Mat>& vSingleMats);

private:
    logger::Logger mGLogger;

	std::string mEnginePath;

	int mClipLen;

	nvinfer1::IExecutionContext* mContext;
	nvinfer1::ICudaEngine* mEngine;
	nvinfer1::IRuntime* mRuntime;

	cudaStream_t mStream;
	void* mBuffers[2];

	float* mBlob;
	float* mProb;
	int mInputIndex;
	int mOutputIndex;
	int mInputSize;
	int mOutputSize;

	float mMeans[3] = { 123.0f, 117.0f, 104.0f };
	float mStd[3] = { 58.395f, 57.12f, 57.375f };

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

	static inline std::pair<int, float> argmax(std::vector<float>& vProbs)
	{
		std::pair<int, float> result;
		auto iter = std::max_element(vProbs.begin(), vProbs.end());
		result.first = static_cast<int>(iter - vProbs.begin());
		result.second = *iter;

		return result;
	}

	static inline float sigmoid(float x)
	{
		return static_cast<float>(1.f / (1.f + exp(-x)));
	}
};

