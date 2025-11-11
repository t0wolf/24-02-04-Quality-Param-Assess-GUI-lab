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

// 自定义日志类，继承自 nvinfer1::ILogger
class SegLogger : public nvinfer1::ILogger {
public:
	// 重写 log 函数，用于处理日志消息
	void log(Severity severity, const char* msg) noexcept override {
		// 根据日志级别输出消息
		switch (severity) {
		case Severity::kINTERNAL_ERROR:
			std::cerr << "[INTERNAL_ERROR] " << msg << std::endl;
			break;
		case Severity::kERROR:
			std::cerr << "[ERROR] " << msg << std::endl;
			break;
		case Severity::kWARNING:
			std::cerr << "[WARNING] " << msg << std::endl;
			break;
		case Severity::kINFO:
			std::cout << "[INFO] " << msg << std::endl;
			break;
		case Severity::kVERBOSE:
			std::cout << "[VERBOSE] " << msg << std::endl;
			break;
		default:
			break;
		}
	}
};

class SegmentInferBase
{
public:
	SegmentInferBase(const std::string& sEngineFilePath);  // 构造函数

	virtual ~SegmentInferBase();  // 析构函数

	int doInference(cv::Mat& src, cv::Mat& mask);  // 执行推理

	int doInference(cv::Mat& src, std::vector<cv::Mat>& vMasks);

protected:
	nvinfer1::Dims4 m_inputDims;
	nvinfer1::Dims4 m_outputDims;
	nvinfer1::ICudaEngine* m_engine;
	nvinfer1::IExecutionContext* m_context;

	std::vector<float> m_means;
	std::vector<float> m_stds;

	int m_inputImageIdx;
	int m_outputMaskIdx;

	int m_classes;

protected:
	int preprocess(cv::Mat& src, float* blob);  // 图像预处理

	int blobFromImage(cv::Mat& src, float* blob);  // 将属兔图像的每一个像素存入数组


	int doSingleInfer(cv::Mat& src, cv::Mat& mask);  // 单张图片推理

	int doSingleInfer(cv::Mat& src, std::vector<cv::Mat>& vMasks);

	int postProcess(float* output, cv::Mat& mask);  // 后处理

	int postProcess(float* output, std::vector<cv::Mat>& vMasks);

	inline size_t getMemorySize(const nvinfer1::Dims& dims, const int32_t elem_size)  // 计算给定维度的数据在内存中所占用的总字节数。
																					  // inline，向编译器提供函数内联展开的建议，将函数的代码插入到调用该函数的地方，而不是通过函数调用的方式执行。
	{
		return std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<int64_t>()) * elem_size;
	}

	inline size_t getElementNum(const nvinfer1::Dims& dims)  // 计算给定维度的元素数量
	{
		return std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<int64_t>());
	}
};

