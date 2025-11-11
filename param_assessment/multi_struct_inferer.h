#pragma once
#include <cassert>
#include <numeric>
#include <cfloat>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>

#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "logger.h"
#include "image_process.h"


class MultiStructInferer
{
public:
	MultiStructInferer(const std::string sEngineFilePath);

	~MultiStructInferer();

	int doInference(cv::Mat& src, std::vector<cv::Mat>& vMasks);

	int doInference(std::vector<cv::Mat>& imgs, std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics);

	int doInference(std::vector<cv::Mat>& imgs, std::vector<cv::Mat>& vMasks);

private:
	int doSingleInfer(std::vector<cv::Mat>& imgs, std::vector<cv::Mat>& vMasks);

	cv::Mat preprocessSingleImage(cv::Mat& src);

	int preprocess(std::vector<cv::Mat>& imgs, float* blob);

	std::vector<cv::Mat> MultiStructInferer::postProcess(float* outputED, float* outputES);

	inline size_t getMemorySize(const nvinfer1::Dims& dims, const int32_t elem_size)
	{
		return std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<int64_t>()) * elem_size;
	}

	inline size_t getElementNum(const nvinfer1::Dims& dims)
	{
		return std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<int64_t>());
	}

protected:
	nvinfer1::Dims4 m_inputDims;
	nvinfer1::Dims4 m_outputEDDims;
	nvinfer1::Dims4 m_outputESDims;
	nvinfer1::ICudaEngine* m_engine;
	nvinfer1::IExecutionContext* m_context;

	std::vector<float> m_means = { 0.485f, 0.456f, 0.406f };
	std::vector<float> m_stds = { 0.229f, 0.224f, 0.225f };

	int m_inputImageIdx;
	int m_outputEDMaskIdx, m_outputESMaskIdx;

	int m_edClasses, m_esClasses;
};

