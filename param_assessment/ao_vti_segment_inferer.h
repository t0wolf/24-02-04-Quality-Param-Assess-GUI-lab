#pragma once

#include "segment_infer_base.h"
#include <numeric>
#include <algorithm>
#include <utility>

class AoVTISegmentInferer
{
public:
	AoVTISegmentInferer(std::string& sEngineFilePath);

	~AoVTISegmentInferer();

	int doInference(cv::Mat& src, std::vector<cv::Mat>& vMasks);

	std::vector<cv::Mat> scaleMasks(const std::vector<cv::Mat>& masks, const cv::Size& originalSize);

	//void postprocess(float* output0, float* output1, cv::Mat src);

	//void postprocess(float* output0, float* output1);

private:
	int doSingleInfer(cv::Mat& src, std::vector<cv::Mat>& vMasks);

	int preprocess(cv::Mat& src, cv::Mat& dst, float* blob);

  //  void detect(std::vector<cv::Mat>& mask,cv::Mat &frame);

	void sigmoidActivation(cv::Mat& mat);

	void drawPred(float conf, int left, int top, int right, int bottom, cv::Mat& frame, int classid);

	cv::Mat crop_mask(const cv::Mat& mask, const cv::Rect& box);

	cv::Mat process_mask(const std::vector<cv::Mat>& output1, const std::vector<float>& mask_real, const cv::Rect& box, const cv::Size& shape, bool upsample);

    void postProcess(std::vector<cv::Mat>& mask, cv::Mat& output1, cv::Mat& frame, std::vector<cv::Mat>& vMasks);

	inline size_t getMemorySize(const nvinfer1::Dims& dims, const int32_t elem_size)  // 计算给定维度的数据在内存中所占用的总字节数。
		// inline，向编译器提供函数内联展开的建议，将函数的代码插入到调用该函数的地方，而不是通过函数调用的方式执行。
	{
		return std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<int64_t>()) * elem_size;
	}

	inline size_t getElementNum(const nvinfer1::Dims& dims)  // 计算给定维度的元素数量
	{
		return std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<int64_t>());
	}

	//int postProcess(float* output, std::vector<cv::Mat>& vMasks);

public:
	nvinfer1::Dims4 m_inputDims;
	nvinfer1::Dims4 m_outputDims1;
	nvinfer1::Dims3 m_outputDims0;
	nvinfer1::ICudaEngine* m_engine;
	nvinfer1::IExecutionContext* m_context;

	std::vector<float> m_means;
	std::vector<float> m_stds;

	int m_inputImageIdx;
	int m_outputIdx1;
	int m_outputIdx0;
	int m_classes;
	const std::vector<std::string> class_names = { "beat","HEART_BEAT" };
};

//class Matrix {
//public:
//    Matrix();
//    Matrix(int rows, int cols, const std::initializer_list<float>& pdata = {});
//    Matrix(int rows, int cols, const std::vector<float>& v);
//
//    const float& operator()(int irow, int icol)const { return data_[irow * cols_ + icol]; }
//    float& operator()(int irow, int icol) { return data_[irow * cols_ + icol]; }
//    Matrix element_wise(const std::function<float(float)>& func) const;
//    Matrix operator*(const Matrix& value) const;
//    Matrix operator*(float value) const;
//    Matrix operator+(float value) const;
//    Matrix operator-(float value) const;
//    Matrix operator/(float value) const;
//    int rows() const { return rows_; }
//    int cols() const { return cols_; }
//    Matrix view(int rows, int cols) const;
//    Matrix power(float y) const;
//    float reduce_sum() const;
//    float* ptr() const { return (float*)data_.data(); }
//    Matrix exp(float value);
//    Matrix mygemm(const Matrix& A, const Matrix& B);
//public:
//    int rows_ = 0;
//    int cols_ = 0;
//    std::vector<float> data_;
//};
