#define __STDC_FORMAT_MACROS
//#include "argsParser.h"
#include "buffers.h"
//#include "common.h"
#include "logger.h"
//#include "parserOnnxConfig.h"
#include "NvInfer.h"
#include <cuda_runtime_api.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <ctime>
#include <math.h>
//#include <opencv.hpp>
#include "opencv2/opencv.hpp"

#include "typemappings.h"

//using namespace sample;
using namespace samplesCommon;
using namespace nvinfer1;
//using namespace nvonnxparser;

//struct InferDeleter
//{
//	template <typename T>
//	void operator()(T* obj) const
//	{
//		delete obj;
//	}
//};

//template <typename T>
//using SampleUniquePtr = std::unique_ptr<T, InferDeleter>;


class AngleInferPLAX {
private:
	cv::Size mInputSize;
	cv::Size mOutputSize;
	int inputWeight, inputHeight;
	int outputWeight, outputHeight;
	float scale;

	int w, h, x, y;
	std::shared_ptr<ICudaEngine> engine;
	SampleUniquePtr<IExecutionContext> context;
	cv::Mat preprocess_img(cv::Mat& img);

public:
	AngleInferPLAX(std::string enginePath, cv::Size mInputSize, cv::Size mOutputSize);
	~AngleInferPLAX();
	void build(std::string engineName);
	int doInference(cv::Mat image);
};