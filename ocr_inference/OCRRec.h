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
#include <QObject>
#include <QDebug>
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "logger.h"
#include "param_assessment/image_process.h"

class OCRRec
{
public:
	OCRRec(std::string sDetModelPath, std::string dictionaryPath);

	~OCRRec();

	int doInference(std::vector<cv::Mat>& vSrc, std::vector<std::string>& vChTexts);

private:
	int parseDict(std::string dictionaryPath);

	int parseChineseDict(std::string& dictionaryPath);

	int preprocess(cv::Mat& src, float* blob);

	int resizeImg(cv::Mat& src, cv::Mat& dst);

	void crnnResizeImg(const cv::Mat& img, cv::Mat& resize_img, float wh_ratio);

	void Normalize(cv::Mat* im,
		const std::vector<float>& mean = { 0.5f, 0.5f, 0.5f },
		const std::vector<float>& scale = { 1 / 0.5f, 1 / 0.5f, 1 / 0.5f },
		const bool is_scale = true)
	{
		double e = 1.0;
		if (is_scale)
		{
			e /= 255.0;
		}
		(*im).convertTo(*im, CV_32FC3, e);
		std::vector<cv::Mat> bgr_channels(3);
		cv::split(*im, bgr_channels);
		for (auto i = 0; i < bgr_channels.size(); i++)
		{
			bgr_channels[i].convertTo(bgr_channels[i], CV_32FC1, 1.0 * scale[i],
				(0.0 - mean[i]) * scale[i]);
		}
		cv::merge(bgr_channels, *im);
	}

	int postProcess(float* output, std::string& vChTexts);

	int blobFromImage(cv::Mat& src, float* blob);

	int blobFromImage(cv::Mat& src, float* blob, bool normalize);

	int doSingleInfer(cv::Mat& src, std::string& vChTexts);

	inline size_t getMemorySize(const nvinfer1::Dims& dims, const int32_t elem_size)
	{
		return std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<int64_t>()) * elem_size;
	}

	inline size_t getElementNum(const nvinfer1::Dims& dims)
	{
		return std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<int64_t>());
	}


private:
	nvinfer1::Dims4 m_inputDims;
	nvinfer1::Dims3 m_outputDims;
	nvinfer1::ICudaEngine* m_engine;
	nvinfer1::IExecutionContext* m_context;

	std::vector<float> m_means = {0.485f, 0.456f, 0.406f};
	std::vector<float> m_stds = {0.229f, 0.224f, 0.225f};
	std::vector<std::string> m_dictionary;

	cv::Size m_originImgSize;

	int m_inputImageIdx;
	int m_outputMaskIdx;
};

