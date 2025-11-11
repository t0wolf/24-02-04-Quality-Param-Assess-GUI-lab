#pragma once
#include <cassert>
#include <numeric>
#include <cfloat>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <QDebug>

#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "logger.h"
#include "param_assessment/image_process.h"


class OCRDet
{
public:
	OCRDet(std::string sDetModelPath);

	~OCRDet();

	int doInference(cv::Mat& src, std::vector<cv::Rect>& vChBoxes);

private:
	int preprocess(cv::Mat& src, float* blob, float& ratio_h, float& ratio_w);

	int preprocess(cv::Mat& src, cv::Mat& dst, float& ratio_h, float& ratio_w);

	int resizeImage(cv::Mat& src, cv::Mat& dst);

	int dbResizeImage(cv::Mat& src, cv::Mat& dst, int max_size_len, float& ratio_h, float& ratio_w);

	int boxesProjBack(cv::Rect& detectedBox);

	int boxesProjBack(cv::Rect& detectedBox, float& ratio_h, float& ratio_w);

	int expandROI(cv::Rect& rect, int expandSize);

	int blobFromImage(cv::Mat& src, float* blob);

	int doSingleInfer(cv::Mat& src, std::vector<cv::Rect>& vChBoxes);

	int postProcess(float* output, nvinfer1::Dims4& outputDims, std::vector<cv::Rect>& vChBoxes, float& ratio_h, float& ratio_w);

	int argmaxMask(float* output, nvinfer1::Dims4& outputDims, cv::Mat& mask);

	int getBoxesFromMask(cv::Mat& mask, std::vector<cv::Rect>& vChBoxes, float& ratio_h, float& ratio_w);

	int getMinimizeBoxes(std::vector<cv::Point>& contours, std::vector<cv::Point>& miniBoxes);

	inline size_t getMemorySize(const nvinfer1::Dims& dims, const int32_t elem_size)
	{
		return std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<int64_t>()) * elem_size;
	}

	inline size_t getElementNum(const nvinfer1::Dims& dims)
	{
		return std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<int64_t>());
	}

	std::vector<std::vector<cv::Point>> filterAbnormalSmallContours(std::vector<std::vector<cv::Point>>& contours);

protected:
	nvinfer1::Dims4 m_inputDims;
	nvinfer1::Dims4 m_outputDims;
	nvinfer1::ICudaEngine* m_engine;
	nvinfer1::IExecutionContext* m_context;

	std::vector<float> m_means = {0.485f, 0.456f, 0.406f};
	std::vector<float> m_stds = { 0.229f, 0.224f, 0.225f };

	cv::Size m_originImgSize;

	int m_inputImageIdx;
	int m_outputMaskIdx0;
	int m_outputMaskIdx1;

	float m_scaleX = 0.0f;
	float m_scaleY = 0.0f;
	int m_offsetX = 0;
	int m_offsetY = 0;
};

