#pragma once
#include "segment_infer_base.h"

class PLAXPoseClsInferer
{
public:
	PLAXPoseClsInferer(std::string& sEngineFilePath);

	~PLAXPoseClsInferer();

	int doInference(cv::Mat& src, std::map<std::string, int>& classResults);

	std::vector<std::string> m_viewNameMap = { "Aortic", "Both", "LV" };

private:
	int doSingleInfer(cv::Mat& src, std::map<std::string, int>& classResults);

	int preprocess(cv::Mat& src, float* blob);

	int blobFromImage(cv::Mat& src, float* blob);

	int postProcess(float* outputView, std::map<std::string, int>& classResults);

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
	nvinfer1::Dims2 m_outputViewDims;
	nvinfer1::ICudaEngine* m_engine;
	nvinfer1::IExecutionContext* m_context;

	std::vector<float> m_means = { 0.485f, 0.456f, 0.406f };
	std::vector<float> m_stds = { 0.229f, 0.224f, 0.225f };

	int m_inputImageIdx;
	int m_outputViewIdx;

	int m_classes;
};

