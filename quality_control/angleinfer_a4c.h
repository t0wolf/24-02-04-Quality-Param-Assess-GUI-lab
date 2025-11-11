//
// Created by Ziyon on 2022/7/12.
//

#define __STDC_FORMAT_MACROS
// #include "common/argsParser.h"
#include "buffers.h"
// #include "common/common.h"
#include "logger.h"
// #include "parserOnnxConfig.h"
#include "NvInfer.h"
#include <cuda_runtime_api.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <ctime>
#include <math.h>
#include <opencv.hpp>

#include "typemappings.h"

// using namespace sample;
using namespace samplesCommon;
using namespace nvinfer1;
// using namespace nvonnxparser;


class AngleInferA4C {
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
	cv::Point postprocess_img(cv::Mat img);
	std::vector<cv::Point> findMaxRegion(std::vector<std::vector<cv::Point>> contours);
	std::vector<int> findPeaks(std::vector<float> num, int count);
public:
	AngleInferA4C(std::string enginePath, cv::Size mInputSize, cv::Size mOutputSize);
	~AngleInferA4C();
	void build(std::string engineName);
	cv::Point doInference(cv::Mat image);
};
