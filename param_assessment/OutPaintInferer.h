#pragma once
#include "segment_infer_base.h"

class OutPaintInferer
{
public:
	OutPaintInferer(std::string& sEngineFilePath);

	~OutPaintInferer();

	int doInference(std::vector<cv::Mat>& video, std::vector<cv::Mat>& outpaint_video);

private:
	int doSingleInfer(std::vector<cv::Mat>& video, std::vector<cv::Mat>& outpaint_video);

	int preprocess(std::vector<cv::Mat>& video, float* blob);

	int blobFromImage(std::vector<cv::Mat>& video, float* blob);

	int postProcess(float* outputEd, std::vector<cv::Mat>& outpaint_video);

	inline size_t getMemorySize(const nvinfer1::Dims& dims, const int32_t elem_size)  // 计算给定维度的数据在内存中所占用的总字节数。
		// inline，向编译器提供函数内联展开的建议，将函数的代码插入到调用该函数的地方，而不是通过函数调用的方式执行。
	{
		return std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<int64_t>()) * elem_size;
	}

	inline size_t getElementNum(const nvinfer1::Dims& dims)  // 计算给定维度的元素数量
	{
		return std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<int64_t>());
	}

protected:
	nvinfer1::Dims4 m_inputDims;
	//nvinfer1::Dims4 m_outputEdDims;
	nvinfer1::Dims4 m_outputEsDims;
	nvinfer1::Dims4 m_outputDims;
	nvinfer1::ICudaEngine* m_engine;
	nvinfer1::IExecutionContext* m_context;
	std::vector<float> m_means;
	std::vector<float> m_stds;

	int m_inputImageIdx;
	int m_outputIdx;
	int m_outputEsIdx;

	int m_classes;

}; 
